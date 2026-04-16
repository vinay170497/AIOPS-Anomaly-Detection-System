"""
ingestion_engine.py
--------------------
Streaming ingestion pipeline for 100M+ server logs.
Uses Drain3 for template mining + DuckDB for disk-backed columnar storage.
All processing is chunked/generator-based to prevent OOM errors.
"""

import os
import re
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Generator, Iterator, Optional

import duckdb
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("IngestionEngine")


# ── Regex patterns for common log formats ──────────────────────────────────────
LOG_PATTERNS = {
    "combined_apache": re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
        r'"(?P<method>\S+)? (?P<path>\S+)? \S+?" '
        r'(?P<status>\d{3}) (?P<size>\S+)'
    ),
    "syslog": re.compile(
        r'(?P<timestamp>\w{3}\s+\d+\s+\d+:\d+:\d+) '
        r'(?P<host>\S+) (?P<service>\S+)\[?(?P<pid>\d*)\]?: '
        r'(?P<message>.+)'
    ),
    "log4j": re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}[,\.]\d+)\s+'
        r'(?P<level>DEBUG|INFO|WARN|ERROR|FATAL|TRACE)\s+'
        r'(?P<logger>\S+)\s+-\s+(?P<message>.+)'
    ),
    "generic": re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\.,]?\d*)\s*'
        r'(?P<level>DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL|TRACE)?\s*'
        r'(?P<message>.+)'
    )
}

HTTP_STATUS_CATEGORIES = {
    "1xx": range(100, 200),
    "2xx": range(200, 300),
    "3xx": range(300, 400),
    "4xx": range(400, 500),
    "5xx": range(500, 600),
}

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS raw_logs (
    log_id          VARCHAR PRIMARY KEY,
    line_number     BIGINT,
    source_file     VARCHAR,
    raw_line        VARCHAR,
    timestamp       TIMESTAMP,
    level           VARCHAR,
    service         VARCHAR,
    template_id     VARCHAR,
    template_str    VARCHAR,
    dynamic_params  VARCHAR,      -- JSON list of extracted parameters
    http_status     INTEGER,
    http_1xx        BOOLEAN DEFAULT FALSE,
    http_2xx        BOOLEAN DEFAULT FALSE,
    http_3xx        BOOLEAN DEFAULT FALSE,
    http_4xx        BOOLEAN DEFAULT FALSE,
    http_5xx        BOOLEAN DEFAULT FALSE,
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_template ON raw_logs(template_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON raw_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_level     ON raw_logs(level);
"""


class LogIngestionEngine:
    """
    Streaming log ingestion with Drain3 template mining and DuckDB persistence.

    Usage:
        engine = LogIngestionEngine(db_path="logs.duckdb")
        engine.ingest_file("server.log", chunk_size=10_000)
    """

    def __init__(
        self,
        db_path: str = "aiops_logs.duckdb",
        config_path: str = "miner_config.json",
        drain_state_path: str = "drain_state.bin",
        chunk_size: int = 5_000,
    ):
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.drain_state_path = drain_state_path

        # ── DuckDB connection ──────────────────────────────────────────────────
        self.con = duckdb.connect(db_path)
        self.con.execute(SCHEMA_DDL)
        logger.info(f"DuckDB initialised -> {db_path}")

        # ── Drain3 template miner ──────────────────────────────────────────────
        self.miner = self._init_drain(config_path)
        logger.info("Drain3 TemplateMiner ready")

    # ── Initialise Drain3 ─────────────────────────────────────────────────────
    def _init_drain(self, config_path: str) -> TemplateMiner:
        """
        Configure Drain3 programmatically from the JSON config file.

        Drain3's TemplateMinerConfig.load() internally uses Python's configparser,
        which only reads INI-format files.  Passing a JSON file to it raises a
        MissingSectionHeaderError.  We therefore read the JSON ourselves and set
        every attribute directly on the config object — no INI file needed.
        """
        from drain3.file_persistence import FilePersistence
        from drain3.masking import MaskingInstruction

        cfg = TemplateMinerConfig()

        # Load settings from JSON if available, otherwise use safe defaults
        if Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                drain_sec = raw.get("drain", {})
                cfg.drain_sim_th       = float(drain_sec.get("sim_th",       0.4))
                cfg.drain_depth        = int(drain_sec.get("depth",           4))
                cfg.drain_max_children = int(drain_sec.get("max_children",   100))
                cfg.drain_max_clusters = int(drain_sec.get("max_clusters",  1024))
                cfg.drain_extra_delimiters = drain_sec.get(
                    "extra_delimiters", ["_", "/", ":", "=", ",", ";", "|"]
                )

                snap_sec = raw.get("snapshot", {})
                cfg.snapshot_interval_minutes = int(
                    snap_sec.get("snapshot_interval_minutes", 10)
                )
                cfg.snapshot_compress_state = bool(
                    snap_sec.get("compress_state", False)
                )

                # Build masking instructions from the JSON list
                masking_instructions = []
                for m in raw.get("masking", []):
                    pattern = m.get("regex_pattern", "")
                    mask    = m.get("mask_with", "<*>")
                    if pattern:
                        masking_instructions.append(
                            MaskingInstruction(pattern, mask)
                        )
                cfg.masking_instructions = masking_instructions
                logger.info(
                    f"Drain3 config loaded from {config_path} "
                    f"({len(masking_instructions)} masking rules)"
                )

            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    f"Could not parse {config_path}: {exc} -- using defaults"
                )
        else:
            logger.info("No miner_config.json found -- using Drain3 defaults")

        # Persistence handler (resumes from saved state if it exists)
        persistence_handler = FilePersistence(self.drain_state_path)
        if Path(self.drain_state_path).exists():
            logger.info(f"Resuming Drain3 state from {self.drain_state_path}")

        return TemplateMiner(persistence_handler=persistence_handler, config=cfg)

    # ── Parse a single raw log line ───────────────────────────────────────────
    def _parse_line(self, line: str) -> dict:
        line = line.strip()
        if not line:
            return {}

        parsed = {
            "raw_line": line[:4096],  # hard cap for safety
            "timestamp": None,
            "level": None,
            "service": None,
            "http_status": None,
        }
        message = line

        for fmt_name, pattern in LOG_PATTERNS.items():
            m = pattern.search(line)
            if m:
                groups = m.groupdict()
                message = groups.get("message", line)
                parsed["level"] = groups.get("level", "UNKNOWN")
                parsed["service"] = groups.get("service", groups.get("logger", "unknown"))

                # Parse timestamp
                ts_str = groups.get("timestamp", "")
                if ts_str:
                    parsed["timestamp"] = self._parse_timestamp(ts_str)

                # HTTP status code
                if "status" in groups and groups["status"]:
                    try:
                        parsed["http_status"] = int(groups["status"])
                    except ValueError:
                        pass
                break

        # Drain3 template extraction
        # add_log_message() return type changed across drain3 versions:
        #   Shape A (dict)  : {"cluster": LogCluster, "change_type": str}
        #                     -- some dict versions return {} on change_type="none"
        #   Shape B (tuple) : (LogCluster, change_type_str)
        #   Shape C (direct): LogCluster object returned directly
        # _extract_cluster() handles all three shapes safely.
        result  = self.miner.add_log_message(message)
        cluster = self._extract_cluster(result)

        if cluster is not None:
            template_str = self._safe_get_template(cluster)
            parsed["template_id"]    = f"E{cluster.cluster_id}"
            parsed["template_str"]   = template_str
            parsed["dynamic_params"] = self._extract_dynamic_params(
                message, template_str
            )
        else:
            parsed["template_id"]    = "E_UNKNOWN"
            parsed["template_str"]   = message[:512]
            parsed["dynamic_params"] = "[]"

        return parsed, message

    @staticmethod
    def _extract_cluster(result):
        """
        Version-agnostic LogCluster extractor.

        Handles every return shape ever produced by drain3.add_log_message():
          Shape A — dict  : result.get("cluster") -- also covers empty-dict edge case
          Shape B — tuple : result[0]
          Shape C — direct: the result itself IS the LogCluster

        Returns a LogCluster object, or None if nothing could be extracted.
        """
        if result is None:
            return None

        # Shape A: dict (most common in drain3 <= 0.9.13)
        if isinstance(result, dict):
            return result.get("cluster")   # returns None if key absent (e.g. {})

        # Shape B: tuple (LogCluster, change_type_str)
        if isinstance(result, tuple):
            cluster = result[0] if result else None
            return cluster if cluster is not None and hasattr(cluster, "cluster_id") else None

        # Shape C: LogCluster returned directly (some 0.9.14+ builds)
        if hasattr(result, "cluster_id"):
            return result

        return None

    @staticmethod
    def _safe_get_template(cluster) -> str:
        """
        Safely extract the template string from a LogCluster object.
        Handles both .get_template() method (0.9.x) and direct
        .log_template_tokens list (all versions).
        """
        try:
            # Preferred: method available in drain3 0.9.x+
            return cluster.get_template()
        except AttributeError:
            pass
        try:
            # Fallback: join the token list directly
            return " ".join(cluster.log_template_tokens)
        except (AttributeError, TypeError):
            pass
        return str(cluster)

    def _count_clusters(self) -> int:
        """
        Return the number of unique templates mined so far.
        Handles internal attribute renames across drain3 versions:
          drain3 <= 0.9.x : self.miner.drain.id_to_cluster  (dict)
          drain3 newer    : self.miner.drain.id_to_cluster or .clusters
        """
        drain = getattr(self.miner, "drain", None)
        if drain is None:
            return 0
        for attr in ("id_to_cluster", "clusters", "id_to_cluster_map"):
            mapping = getattr(drain, attr, None)
            if mapping is not None:
                return len(mapping)
        return 0

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        FORMATS = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S,%f",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%b %d %H:%M:%S",
            "%d/%b/%Y:%H:%M:%S %z",
        ]
        for fmt in FORMATS:
            try:
                return datetime.strptime(ts_str.strip(), fmt)
            except (ValueError, TypeError):
                continue
        return None

    def _extract_dynamic_params(self, message: str, template: str) -> str:
        """
        Very lightweight dynamic parameter extraction:
        Tokenise both message and template, return tokens in message
        that are replaced by wildcards in template.
        """
        msg_tokens = message.split()
        tmpl_tokens = template.split()
        params = []
        for mt, tt in zip(msg_tokens, tmpl_tokens):
            if tt == "<*>" and mt != "<*>":
                params.append(mt)
        # Also grab any trailing tokens beyond template length
        if len(msg_tokens) > len(tmpl_tokens):
            params.extend(msg_tokens[len(tmpl_tokens):])
        return json.dumps(params[:32])  # cap at 32 params

    def _encode_http_flags(self, status: Optional[int]) -> dict:
        flags = {k: False for k in ["http_1xx", "http_2xx", "http_3xx", "http_4xx", "http_5xx"]}
        if status is None:
            return flags
        for key, rng in HTTP_STATUS_CATEGORIES.items():
            col = f"http_{key}"
            if status in rng:
                flags[col] = True
        return flags

    def _make_log_id(self, source: str, line_no: int) -> str:
        raw = f"{source}:{line_no}:{time.time_ns()}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    # ── Core generator: yields rows in batches ────────────────────────────────
    def _line_reader(self, filepath: str) -> Generator[str, None, None]:
        """Memory-safe line-by-line reader."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line

    def _batch_generator(
        self, filepath: str, start_line: int = 0
    ) -> Generator[list, None, None]:
        """Yields list-of-dicts in `chunk_size` batches."""
        batch = []
        source = Path(filepath).name
        for line_no, raw_line in enumerate(self._line_reader(filepath), start=1):
            if line_no <= start_line:
                continue
            result = self._parse_line(raw_line)
            if not result:
                continue
            parsed, _ = result

            log_id = self._make_log_id(source, line_no)
            http_flags = self._encode_http_flags(parsed.get("http_status"))

            row = {
                "log_id": log_id,
                "line_number": line_no,
                "source_file": source,
                "raw_line": parsed.get("raw_line", raw_line[:4096]),
                "timestamp": parsed.get("timestamp"),
                "level": parsed.get("level", "UNKNOWN"),
                "service": parsed.get("service", "unknown"),
                "template_id": parsed.get("template_id", "E_UNKNOWN"),
                "template_str": parsed.get("template_str", ""),
                "dynamic_params": parsed.get("dynamic_params", "[]"),
                "http_status": parsed.get("http_status"),
                **http_flags,
            }
            batch.append(row)

            if len(batch) >= self.chunk_size:
                yield batch
                batch = []

        if batch:
            yield batch

    # ── Insert batch into DuckDB ──────────────────────────────────────────────
    def _insert_batch(self, batch: list) -> int:
        if not batch:
            return 0
        # Convert to columnar lists for bulk insert
        columns = list(batch[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        col_names = ", ".join(columns)
        sql = f"INSERT OR IGNORE INTO raw_logs ({col_names}) VALUES ({placeholders})"

        rows = [tuple(row[c] for c in columns) for row in batch]
        self.con.executemany(sql, rows)
        return len(rows)

    # ── Public API ─────────────────────────────────────────────────────────────
    def ingest_file(self, filepath: str, resume: bool = True) -> dict:
        """
        Ingest a log file with streaming chunks.

        Args:
            filepath: Path to the log file.
            resume:   If True, skip lines already ingested (checks max line_number).

        Returns:
            dict with ingestion statistics.
        """
        filepath = str(Path(filepath).resolve())
        source = Path(filepath).name

        start_line = 0
        if resume:
            row = self.con.execute(
                "SELECT MAX(line_number) FROM raw_logs WHERE source_file = ?", [source]
            ).fetchone()
            if row and row[0]:
                start_line = row[0]
                logger.info(f"Resuming from line {start_line}")

        total_rows = 0
        total_chunks = 0
        t0 = time.time()

        for chunk in self._batch_generator(filepath, start_line):
            inserted = self._insert_batch(chunk)
            total_rows += inserted
            total_chunks += 1
            if total_chunks % 10 == 0:
                elapsed = time.time() - t0
                rate = total_rows / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Chunk {total_chunks:,} | Rows: {total_rows:,} | "
                    f"Rate: {rate:,.0f} rows/s"
                )

        # Persist Drain3 state
        self.miner.save_state(self.drain_state_path)
        elapsed = time.time() - t0

        stats = {
            "source_file": source,
            "total_rows_ingested": total_rows,
            "total_chunks": total_chunks,
            "elapsed_seconds": round(elapsed, 2),
            "rows_per_second": round(total_rows / elapsed, 0) if elapsed > 0 else 0,
            "unique_templates": self._count_clusters(),
        }
        logger.info(f"Ingestion complete: {stats}")
        return stats

    def ingest_directory(self, dir_path: str, pattern: str = "*.log") -> list:
        """Ingest all matching log files in a directory."""
        results = []
        for fp in sorted(Path(dir_path).glob(pattern)):
            logger.info(f"Ingesting: {fp}")
            results.append(self.ingest_file(str(fp)))
        return results

    # ── Streaming query interface for downstream phases ───────────────────────
    def stream_feature_rows(
        self,
        batch_size: int = 10_000,
        where_clause: str = "1=1",
        offset: int = 0,
    ) -> Generator[list, None, None]:
        """
        Generator yielding batches of rows for feature extraction.
        Avoids pulling 100M rows into RAM at once.
        """
        total = self.con.execute(
            f"SELECT COUNT(*) FROM raw_logs WHERE {where_clause}"
        ).fetchone()[0]

        logger.info(f"Streaming {total:,} rows in batches of {batch_size:,}")
        current_offset = offset

        while current_offset < total:
            rows = self.con.execute(
                f"""
                SELECT log_id, timestamp, level, service, template_id,
                       template_str, dynamic_params, raw_line,
                       http_status, http_1xx, http_2xx, http_3xx, http_4xx, http_5xx
                FROM raw_logs
                WHERE {where_clause}
                ORDER BY timestamp NULLS LAST, line_number
                LIMIT {batch_size} OFFSET {current_offset}
                """
            ).fetchall()

            if not rows:
                break

            yield rows
            current_offset += len(rows)

    def get_template_statistics(self) -> list:
        """Return aggregate stats per template for EDA."""
        return self.con.execute("""
            SELECT
                template_id,
                template_str,
                COUNT(*) AS frequency,
                COUNT(DISTINCT service) AS unique_services,
                SUM(CASE WHEN http_5xx THEN 1 ELSE 0 END) AS error_5xx_count,
                SUM(CASE WHEN http_4xx THEN 1 ELSE 0 END) AS error_4xx_count
            FROM raw_logs
            GROUP BY template_id, template_str
            ORDER BY frequency DESC
        """).fetchall()

    def get_summary(self) -> dict:
        """Return overall DB summary statistics."""
        row = self.con.execute("""
            SELECT
                COUNT(*)           AS total_logs,
                COUNT(DISTINCT template_id) AS unique_templates,
                COUNT(DISTINCT service)     AS unique_services,
                MIN(timestamp)              AS earliest,
                MAX(timestamp)              AS latest
            FROM raw_logs
        """).fetchone()
        return {
            "total_logs": row[0],
            "unique_templates": row[1],
            "unique_services": row[2],
            "earliest_log": str(row[3]),
            "latest_log": str(row[4]),
        }

    def close(self):
        self.con.close()
        logger.info("DuckDB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestion_engine.py <logfile_or_directory>")
        sys.exit(1)

    path = sys.argv[1]
    with LogIngestionEngine(
        db_path="aiops_logs.duckdb",
        config_path="miner_config.json",
        chunk_size=10_000,
    ) as engine:
        if Path(path).is_dir():
            results = engine.ingest_directory(path)
        else:
            results = [engine.ingest_file(path)]

        print("\n=== Ingestion Summary ===")
        for r in results:
            for k, v in r.items():
                print(f"  {k}: {v}")
        print("\n=== DB Summary ===")
        for k, v in engine.get_summary().items():
            print(f"  {k}: {v}")
