"""
main.py
--------
Single entry point for the Self-Evolving AIOps Log Intelligence System.

FIRST-TIME SETUP (from scratch):
    python main.py --mode all --big data/log_files.csv --small data/logging_monitoring_anomalies.csv

ALL MODES:
    # -- Data ingestion --
    python main.py --mode all          --big data/log_files.csv --small data/logging_monitoring_anomalies.csv
    python main.py --mode ingest-big   --big data/log_files.csv
    python main.py --mode ingest-small --small data/logging_monitoring_anomalies.csv
    python main.py --mode ingest       --input /var/log/app.log
    python main.py --mode ingest       --input reports/incident.pdf
    python main.py --mode ingest       --input data/metrics.xlsx
    python main.py --mode ingest       --input data/events.csv
    python main.py --mode ingest       --input /var/log/               # whole directory

    # -- Model training --
    python main.py --mode train
    python main.py --mode propagate

    # -- Analysis & reporting --
    python main.py --mode analyse --input data/new_logs.log  --query "Why are 5xx errors spiking?"
    python main.py --mode analyse --input data/new_report.pdf
    python main.py --mode analyse --input data/new_events.xlsx
    python main.py --mode chat

    # -- Ops --
    python main.py --mode status

SUPPORTED INPUT FORMATS (--mode ingest and --mode analyse):
    .log / .txt     Plain text — one log entry per line (original behaviour)
    .csv            Auto-detected: Apache Combined Log (no header) or structured (has header)
    .xlsx / .xls    Excel workbooks — each row becomes one log line
    .pdf            PDF documents — text extracted page-by-page, each paragraph = one log line
    directory       All files in the directory processed in alphabetical order

PROJECT-SPECIFIC DATA FILES:
    log_files.csv                     Apache Combined Log, NO header, 11 positional columns
    logging_monitoring_anomalies.csv  Structured anomaly telemetry, HAS header, 31 columns
"""

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging — UTF-8 forced on stream handler (safe on Windows cp1252) ─────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler("aiops.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("AIOps.Main")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "db_path":            "aiops_logs.duckdb",
    "model_dir":          "models",
    "chroma_dir":         "chromadb_store",
    "report_dir":         "reports",
    "drain_config":       "miner_config.json",
    "drain_state":        "drain_state.bin",
    "tmp_dir":            "data",
    "chunk_size":         10_000,
    "embed_batch_size":   256,
    "ae1_epochs":         30,
    "ae2_epochs":         40,
    "train_batch_size":   512,
    "if_contamination":   0.05,
    "anomaly_percentile": 95.0,
    "knn_k":              5,
    "label_confidence":   0.6,
    "ollama_model":       "deepseek-r1:8b",
    "ollama_url":         "http://localhost:11434",
}


def load_config(config_path: str) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            cfg.update(json.load(f))
        logger.info(f"Config loaded from {config_path}")
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# APACHE CSV CONSTANTS  (log_files.csv — positional, no header)
# ══════════════════════════════════════════════════════════════════════════════
_A_IP   = 0;  _A_USER = 2;  _A_TS1  = 3;  _A_TZ   = 4
_A_REQ  = 5;  _A_STAT = 6;  _A_BYTE = 7;  _A_RMS  = 10
_APACHE_TS_FMT = "[%d/%b/%Y:%H:%M:%S %z]"


def _apache_status_to_level(s: int) -> str:
    if s < 400:  return "INFO"
    if s < 500:  return "WARN"
    return "ERROR"


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED ANOMALY CSV CONSTANTS  (logging_monitoring_anomalies.csv)
# ══════════════════════════════════════════════════════════════════════════════
_S_TS  = "Timestamp";  _S_LABEL = "Anomaly_Type"; _S_SEV  = "Severity"
_S_SRC = "Source";     _S_HOST  = "Host_IP";       _S_STAT = "Status"
_S_RMS = "Response_Time_ms";  _S_CPU = "CPU_Usage_Percent"
_S_MEM = "Memory_Usage_MB";   _S_DSK = "Disk_Usage_Percent"
_S_ERR = "Error_Code"; _S_SVCT  = "Service_Type";  _S_AID  = "Anomaly_ID"
_SEV_TO_LEVEL = {"low": "INFO", "medium": "WARN", "high": "ERROR", "critical": "FATAL"}


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL FILE ROUTER
# Converts any supported format into a plain .log file that the existing
# LogIngestionEngine pipeline can consume without modification.
# ══════════════════════════════════════════════════════════════════════════════

# Supported extensions and what they map to
_FORMAT_MAP = {
    ".log":  "log",
    ".txt":  "log",
    ".csv":  "csv",
    ".tsv":  "csv",
    ".xlsx": "excel",
    ".xls":  "excel",
    ".xlsm": "excel",
    ".pdf":  "pdf",
}


def _detect_format(path: str) -> str:
    """Return format string for the given file path."""
    ext = Path(path).suffix.lower()
    return _FORMAT_MAP.get(ext, "log")   # default: treat as plain log


def _route_file_to_log(file_path: str, cfg: dict) -> str:
    """
    Universal converter: any supported format -> plain .log file.

    Decision tree:
      .log / .txt -> copy as-is (or return path directly if already in place)
      .csv / .tsv -> detect Apache vs structured vs generic, convert rows to lines
      .xlsx / .xls -> read with pandas, each row becomes one log line
      .pdf         -> extract text with pdfplumber, paragraphs become log lines

    Returns the path to a .log file ready for LogIngestionEngine.
    Raises ValueError for unsupported formats with a helpful message.
    """
    p      = Path(file_path).resolve()
    fmt    = _detect_format(str(p))
    tmpdir = Path(cfg["tmp_dir"])
    tmpdir.mkdir(exist_ok=True)

    stem    = p.stem
    out_log = str(tmpdir / f"_routed_{stem}.log")

    if fmt == "log":
        # Already the right format — copy to tmp so source_file tracking works
        shutil.copy2(str(p), out_log)
        logger.info(f"[router] Plain log/txt -> {out_log}")
        return out_log

    elif fmt == "csv":
        return _csv_to_log(str(p), out_log, cfg)

    elif fmt == "excel":
        return _excel_to_log(str(p), out_log, cfg)

    elif fmt == "pdf":
        return _pdf_to_log(str(p), out_log)

    else:
        raise ValueError(
            f"Unsupported file format: {p.suffix}\n"
            f"Supported: {', '.join(_FORMAT_MAP.keys())}"
        )


def _csv_to_log(csv_path: str, out_log: str, cfg: dict) -> str:
    """
    Convert any CSV file to a plain .log file.

    Auto-detection order:
      1. If column 0 of row 0 looks like an IP  -> Apache Combined Log (no header)
      2. If headers contain 'Anomaly_Type'       -> structured anomaly telemetry
      3. Otherwise                               -> generic CSV (concat string columns)
    """
    # Peek at the first row to decide format
    try:
        peek = pd.read_csv(
            csv_path, header=None, nrows=1,
            sep=None, engine="python", dtype=str, on_bad_lines="skip",
        )
        first_val = str(peek.iloc[0, 0]).strip()
        is_apache = bool(re.match(r"\d+\.\d+\.\d+\.\d+", first_val))
    except Exception:
        is_apache = False

    if is_apache:
        logger.info(f"[router] CSV detected as Apache Combined Log: {csv_path}")
        return _apache_csv_to_log(csv_path, out_log, cfg)

    # Check headers
    try:
        headers = pd.read_csv(csv_path, nrows=0, dtype=str).columns.tolist()
        is_structured = _S_LABEL in headers
    except Exception:
        is_structured = False

    if is_structured:
        logger.info(f"[router] CSV detected as structured anomaly telemetry: {csv_path}")
        return _structured_csv_to_log(csv_path, out_log, cfg)

    logger.info(f"[router] CSV detected as generic format: {csv_path}")
    return _generic_csv_to_log(csv_path, out_log, cfg)


def _apache_csv_to_log(csv_path: str, out_log: str, cfg: dict) -> str:
    """Convert Apache Combined Log CSV (no header, 11 columns) to .log lines."""
    total_read = total_written = 0
    t0 = time.time()

    with open(out_log, "w", encoding="utf-8") as fout:
        reader = pd.read_csv(
            csv_path, header=None, sep=None, engine="python",
            on_bad_lines="skip", dtype=str, chunksize=cfg["chunk_size"],
            quoting=csv.QUOTE_MINIMAL,
        )
        for chunk_df in reader:
            for _, row in chunk_df.iterrows():
                total_read += 1
                try:
                    username = str(row.iloc[_A_USER]).strip()
                    ts_full  = f"{str(row.iloc[_A_TS1]).strip()} {str(row.iloc[_A_TZ]).strip()}"
                    request  = str(row.iloc[_A_REQ]).strip()
                    status   = str(row.iloc[_A_STAT]).strip()
                    bytes_s  = str(row.iloc[_A_BYTE]).strip()
                    resp_ms  = str(row.iloc[_A_RMS]).strip() if len(row) > _A_RMS else "0"
                    try:
                        iso_ts = datetime.strptime(ts_full, _APACHE_TS_FMT).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        iso_ts = ts_full
                    try:
                        level = _apache_status_to_level(int(status))
                    except ValueError:
                        level = "INFO"
                    req_parts = request.split()
                    method = req_parts[0] if req_parts else "UNKNOWN"
                    path_r = req_parts[1] if len(req_parts) > 1 else "/"
                    fout.write(
                        f"{iso_ts} {level} {username} "
                        f"{method} {path_r} status={status} bytes={bytes_s} rt={resp_ms}ms\n"
                    )
                    total_written += 1
                except (IndexError, AttributeError):
                    continue

            if total_read % 500_000 == 0 and total_read > 0:
                logger.info(f"  Converted {total_read:,} Apache rows")

    logger.info(
        f"[router] Apache CSV: {total_read:,} read, {total_written:,} written "
        f"in {time.time()-t0:.1f}s -> {out_log}"
    )
    return out_log


def _structured_csv_to_log(csv_path: str, out_log: str, cfg: dict) -> str:
    """Convert structured anomaly CSV (with header) to .log lines."""
    total = 0
    with open(out_log, "w", encoding="utf-8") as fout:
        for chunk_df in pd.read_csv(csv_path, chunksize=cfg["chunk_size"], dtype=str):
            chunk_df = chunk_df.fillna("")
            for _, row in chunk_df.iterrows():
                try:
                    ts      = row.get(_S_TS,    "").strip()
                    source  = row.get(_S_SRC,   "unknown").strip()
                    sev     = row.get(_S_SEV,   "Low").strip().lower()
                    label   = row.get(_S_LABEL, "Unknown").strip()
                    status  = row.get(_S_STAT,  "Open").strip()
                    host    = row.get(_S_HOST,  "").strip()
                    resp_ms = row.get(_S_RMS,   "0").strip()
                    cpu     = row.get(_S_CPU,   "0").strip()
                    mem     = row.get(_S_MEM,   "0").strip()
                    disk    = row.get(_S_DSK,   "0").strip()
                    err     = row.get(_S_ERR,   "0").strip()
                    svct    = row.get(_S_SVCT,  "").strip()
                    aid     = row.get(_S_AID,   "").strip()
                    level   = _SEV_TO_LEVEL.get(sev, "INFO")
                    fout.write(
                        f"{ts} {level} {source} anomaly_type={label} status={status} "
                        f"host={host} error_code={err} rt={resp_ms}ms cpu={cpu}% "
                        f"mem={mem}MB disk={disk}% svc={svct} id={aid}\n"
                    )
                    total += 1
                except AttributeError:
                    continue
    logger.info(f"[router] Structured CSV: {total:,} rows -> {out_log}")
    return out_log


def _generic_csv_to_log(csv_path: str, out_log: str, cfg: dict) -> str:
    """
    Convert any unknown-schema CSV to .log lines.

    Strategy: detect all string/object columns, join their values into one log
    line per row. Numeric columns contribute as key=value pairs if present.
    A timestamp column is prepended if detected by name.
    """
    TS_NAMES  = {"timestamp", "time", "datetime", "date", "ts", "created_at", "event_time"}
    LVL_NAMES = {"level", "severity", "log_level", "loglevel", "priority"}
    SVC_NAMES = {"service", "source", "host", "component", "module", "logger"}

    total = 0
    with open(out_log, "w", encoding="utf-8") as fout:
        for chunk_df in pd.read_csv(
            csv_path, chunksize=cfg["chunk_size"], dtype=str, on_bad_lines="skip"
        ):
            chunk_df = chunk_df.fillna("")
            cols = list(chunk_df.columns)
            col_lower = {c: c.lower() for c in cols}

            ts_col  = next((c for c in cols if col_lower[c] in TS_NAMES),  None)
            lvl_col = next((c for c in cols if col_lower[c] in LVL_NAMES), None)
            svc_col = next((c for c in cols if col_lower[c] in SVC_NAMES), None)

            for _, row in chunk_df.iterrows():
                ts    = str(row[ts_col]).strip()  if ts_col  else ""
                level = str(row[lvl_col]).strip() if lvl_col else "INFO"
                svc   = str(row[svc_col]).strip() if svc_col else "unknown"

                # Build message from remaining columns
                skip = {ts_col, lvl_col, svc_col}
                parts = []
                for c in cols:
                    if c in skip:
                        continue
                    v = str(row[c]).strip()
                    if v and v not in ("-", "nan", "None", ""):
                        parts.append(f"{c}={v}" if len(v) < 80 else v[:200])

                msg = " ".join(parts)
                fout.write(f"{ts} {level} {svc} {msg}\n")
                total += 1

    logger.info(f"[router] Generic CSV: {total:,} rows -> {out_log}")
    return out_log


def _excel_to_log(excel_path: str, out_log: str, cfg: dict) -> str:
    """
    Convert Excel workbook to .log lines.

    Each sheet is processed. Rows are treated the same as generic CSV rows:
    string columns are concatenated, timestamp/level/service columns auto-detected.
    All sheets are appended to the same output file.
    """
    TS_NAMES  = {"timestamp", "time", "datetime", "date", "ts", "created_at"}
    LVL_NAMES = {"level", "severity", "log_level", "priority"}
    SVC_NAMES = {"service", "source", "host", "component", "module", "logger"}

    total = 0
    try:
        xl = pd.ExcelFile(excel_path, engine="openpyxl")
    except Exception as e:
        logger.warning(f"[router] Excel open failed: {e} -- trying xlrd engine")
        xl = pd.ExcelFile(excel_path, engine="xlrd")

    with open(out_log, "w", encoding="utf-8") as fout:
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name, dtype=str).fillna("")
            except Exception as e:
                logger.warning(f"[router] Skipping sheet '{sheet_name}': {e}")
                continue

            cols      = list(df.columns)
            col_lower = {c: c.lower().strip() for c in cols}
            ts_col    = next((c for c in cols if col_lower[c] in TS_NAMES),  None)
            lvl_col   = next((c for c in cols if col_lower[c] in LVL_NAMES), None)
            svc_col   = next((c for c in cols if col_lower[c] in SVC_NAMES), None)

            for _, row in df.iterrows():
                ts    = str(row[ts_col]).strip()  if ts_col  else ""
                level = str(row[lvl_col]).strip() if lvl_col else "INFO"
                svc   = str(row[svc_col]).strip() if svc_col else f"excel_{sheet_name}"

                skip  = {ts_col, lvl_col, svc_col}
                parts = []
                for c in cols:
                    if c in skip:
                        continue
                    v = str(row[c]).strip()
                    if v and v not in ("-", "nan", "None", ""):
                        parts.append(f"{c}={v[:80]}" if "=" not in c else v[:80])

                fout.write(f"{ts} {level} {svc} {' '.join(parts)}\n")
                total += 1

            logger.info(f"[router] Excel sheet '{sheet_name}': {len(df):,} rows")

    logger.info(f"[router] Excel total: {total:,} rows -> {out_log}")
    return out_log


def _pdf_to_log(pdf_path: str, out_log: str) -> str:
    """
    Extract text from a PDF and write each non-empty paragraph as one log line.

    Uses pdfplumber (already in requirements.txt). Falls back gracefully with
    a clear message if pdfplumber is unavailable.

    Limitations:
      - Scanned PDFs (image-only) will produce no text; use OCR first.
      - Multi-column layouts may produce garbled line order.
      - Tables are extracted as text rows.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF ingestion.\n"
            "Install it with: pip install pdfplumber"
        )

    total_lines = 0
    total_pages = 0

    with open(out_log, "w", encoding="utf-8") as fout, \
         pdfplumber.open(pdf_path) as pdf:

        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Split into paragraphs/lines, skip blank lines
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                # Synthesise a structured log line with page metadata
                fout.write(f"pdf_page={page_num} INFO pdf_source {line[:512]}\n")
                total_lines += 1

            # Also extract tables as structured log lines
            for table in page.extract_tables() or []:
                for row in table:
                    if not row:
                        continue
                    cells = [str(c).strip() for c in row if c and str(c).strip()]
                    if cells:
                        row_text = " | ".join(cells)
                        fout.write(
                            f"pdf_page={page_num} INFO pdf_table {row_text[:512]}\n"
                        )
                        total_lines += 1

    logger.info(
        f"[router] PDF: {total_pages} pages, {total_lines:,} lines -> {out_log}"
    )
    return out_log


def _route_directory(dir_path: str, cfg: dict) -> list:
    """
    Route all supported files in a directory.
    Returns list of converted .log file paths.
    """
    p = Path(dir_path)
    supported = set(_FORMAT_MAP.keys())
    files = sorted(
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in supported
        and not f.name.startswith("_")   # skip temp files
    )
    if not files:
        logger.warning(f"[router] No supported files found in {dir_path}")
        return []

    log_paths = []
    for f in files:
        try:
            log_path = _route_file_to_log(str(f), cfg)
            log_paths.append(log_path)
            logger.info(f"[router] Processed: {f.name}")
        except Exception as e:
            logger.warning(f"[router] Skipping {f.name}: {e}")
    return log_paths


# ══════════════════════════════════════════════════════════════════════════════
# MODE: INGEST-BIG  (Apache Combined Log CSV — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def run_ingest_big(cfg: dict, csv_path: str):
    """Ingest log_files.csv (Apache Combined Log, no header) into DuckDB."""
    from ingestion_engine import LogIngestionEngine

    tmp_log = str(Path(cfg["tmp_dir"]) / "_big_converted.log")
    Path(cfg["tmp_dir"]).mkdir(exist_ok=True)
    logger.info(f"[ingest-big] {csv_path}")

    _apache_csv_to_log(csv_path, tmp_log, cfg)

    logger.info("[ingest-big] Feeding converted log into DuckDB ...")
    with LogIngestionEngine(
        db_path=cfg["db_path"],
        config_path=cfg["drain_config"],
        drain_state_path=cfg["drain_state"],
        chunk_size=cfg["chunk_size"],
    ) as engine:
        stats   = engine.ingest_file(tmp_log, resume=True)
        summary = engine.get_summary()

    _print_section("INGEST-BIG COMPLETE")
    for k, v in stats.items():
        print(f"  {k:<30}: {v}")
    print()
    for k, v in summary.items():
        print(f"  {k:<30}: {v}")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# MODE: INGEST-SMALL  (structured anomaly CSV — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def run_ingest_small(cfg: dict, csv_path: str):
    """
    Two-pass processing of logging_monitoring_anomalies.csv:
    Pass A -> DuckDB ingestion, Pass B -> ChromaDB label embedding.
    """
    from ingestion_engine import LogIngestionEngine
    from knowledge_manager import KnowledgeManager
    from sentence_transformers import SentenceTransformer

    Path(cfg["tmp_dir"]).mkdir(exist_ok=True)
    tmp_log = str(Path(cfg["tmp_dir"]) / "_small_converted.log")
    logger.info(f"[ingest-small] {csv_path}")

    # Pass A
    _structured_csv_to_log(csv_path, tmp_log, cfg)
    with LogIngestionEngine(
        db_path=cfg["db_path"],
        config_path=cfg["drain_config"],
        drain_state_path=cfg["drain_state"],
        chunk_size=cfg["chunk_size"],
    ) as engine:
        stats_a = engine.ingest_file(tmp_log, resume=False)
    logger.info(f"[ingest-small] Pass A - DuckDB stats: {stats_a}")

    # Pass B — embed + push labels
    logger.info("[ingest-small] Pass B - Embedding into ChromaDB ...")
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    km         = KnowledgeManager(persist_dir=cfg["chroma_dir"])
    records    = []
    total_b    = skipped = 0
    label_dist: dict = {}

    for chunk_df in pd.read_csv(csv_path, chunksize=cfg["chunk_size"], dtype=str):
        chunk_df = chunk_df.fillna("")
        for _, row in chunk_df.iterrows():
            label  = row.get(_S_LABEL, "").strip()
            source = row.get(_S_SRC,   "unknown").strip()
            sev    = row.get(_S_SEV,   "Low").strip().lower()
            aid    = row.get(_S_AID,   str(total_b)).strip()
            resp   = row.get(_S_RMS,   "0").strip()
            cpu    = row.get(_S_CPU,   "0").strip()
            mem    = row.get(_S_MEM,   "0").strip()
            disk   = row.get(_S_DSK,   "0").strip()
            err    = row.get(_S_ERR,   "0").strip()
            svct   = row.get(_S_SVCT,  "").strip()

            if not label:
                skipped += 1
                continue

            label_dist[label] = label_dist.get(label, 0) + 1
            embed_text = (
                f"{label} anomaly on {source} service. "
                f"Severity={sev}. ErrorCode={err}. "
                f"ResponseTime={resp}ms. CPU={cpu}% Memory={mem}MB "
                f"Disk={disk}% ServiceType={svct}"
            )[:512]

            def _f(v):
                try: return float(v)
                except (ValueError, TypeError): return 0.0

            records.append({
                "template_id":    f"ANOMALY_{aid}",
                "template_str":   embed_text,
                "_embed_text":    embed_text,
                "frequency":      1,
                "avg_entropy":    0.0,
                "avg_burstiness": 0.0,
                "avg_volatility": _f(resp),
                "avg_time_delta": 0.0,
                "error_rate":     1.0 if (sev in ("high","critical") and _f(err)>0) else 0.0,
                "is_golden":      False,
                "label":          label,
            })
            total_b += 1

            if len(records) >= 500:
                _embed_and_upsert(records, embedder, km)
                records = []
                if total_b % 10_000 == 0:
                    logger.info(f"  Embedded {total_b:,} records...")

    if records:
        _embed_and_upsert(records, embedder, km)

    logger.info(f"[ingest-small] Pass B - {total_b:,} records embedded, {skipped} skipped")

    _print_section("INGEST-SMALL COMPLETE")
    print(f"  DuckDB rows ingested : {stats_a.get('total_rows_ingested',0):,}")
    print(f"  ChromaDB records     : {total_b:,}")
    print(f"  Label types found    : {len(label_dist)}")
    for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"    {lbl:<30}: {cnt:,}")
    return {"ingested": stats_a.get("total_rows_ingested", 0), "embedded": total_b, "labels": label_dist}


def _embed_and_upsert(records: list, embedder, km):
    texts = [r.pop("_embed_text") for r in records]
    embs  = embedder.encode(
        texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True
    ).astype(np.float32)
    for rec, emb in zip(records, embs):
        rec["embedding"] = emb
    km.bulk_upsert_templates(records)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: INGEST  (any supported format — extended with universal router)
# ══════════════════════════════════════════════════════════════════════════════
def run_ingest(cfg: dict, input_path: str):
    """
    Ingest any supported file format into DuckDB.
    Supported: .log, .txt, .csv, .xlsx, .xls, .pdf, or a directory.

    The file is converted to a plain .log file first via the universal router,
    then fed into LogIngestionEngine exactly as before.
    """
    from ingestion_engine import LogIngestionEngine

    p = Path(input_path)
    logger.info(f"[ingest] {input_path}  (format: {_detect_format(str(p))})")

    if p.is_dir():
        # Directory: route all supported files, then ingest each
        log_paths = _route_directory(str(p), cfg)
        if not log_paths:
            logger.error("[ingest] No supported files found in directory")
            return
        with LogIngestionEngine(
            db_path=cfg["db_path"],
            config_path=cfg["drain_config"],
            drain_state_path=cfg["drain_state"],
            chunk_size=cfg["chunk_size"],
        ) as engine:
            all_stats = []
            for lp in log_paths:
                stats = engine.ingest_file(lp, resume=True)
                all_stats.append(stats)
            summary = engine.get_summary()

        _print_section("INGEST COMPLETE")
        for s in all_stats:
            print(f"  {s.get('source_file','?'):<35}: {s.get('total_rows_ingested',0):,} rows")
        print()
        for k, v in summary.items():
            print(f"  {k:<30}: {v}")
        return

    # Single file: route to .log then ingest
    log_path = _route_file_to_log(str(p), cfg)

    with LogIngestionEngine(
        db_path=cfg["db_path"],
        config_path=cfg["drain_config"],
        drain_state_path=cfg["drain_state"],
        chunk_size=cfg["chunk_size"],
    ) as engine:
        stats   = engine.ingest_file(log_path, resume=True)
        summary = engine.get_summary()

    _print_section("INGESTION COMPLETE")
    for k, v in stats.items():
        print(f"  {k:<30}: {v}")
    print()
    print("  DATABASE SUMMARY")
    for k, v in summary.items():
        print(f"  {k:<30}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: TRAIN  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def run_train(cfg: dict, max_rows: int = None):
    """Train AE1 -> IF -> AE2 ensemble on all ingested DuckDB logs."""
    from ingestion_engine import LogIngestionEngine
    from feature_extractor import FeatureExtractor
    from anomaly_pipeline import AnomalyPipeline
    from knowledge_manager import KnowledgeManager

    logger.info("=== TRAINING MODE ===")
    t_start = time.time()

    logger.info("Phase 1/4: Streaming feature extraction from DuckDB")
    engine = LogIngestionEngine(
        db_path=cfg["db_path"],
        config_path=cfg["drain_config"],
        drain_state_path=cfg["drain_state"],
        chunk_size=cfg["chunk_size"],
    )
    fe = FeatureExtractor(batch_embed_size=cfg["embed_batch_size"])

    all_features, all_metadata, total_rows = [], [], 0

    for features, metadata in fe.stream_features(
        engine.stream_feature_rows(batch_size=cfg["chunk_size"])
    ):
        all_features.append(features)
        all_metadata.extend(metadata)
        total_rows += len(metadata)
        if max_rows and total_rows >= max_rows:
            logger.info(f"  max_rows cap reached: {max_rows:,}")
            break
        if total_rows % 100_000 == 0:
            logger.info(f"  Extracted {total_rows:,} feature vectors")

    engine.close()

    if not all_features:
        logger.error("No features extracted. Run ingest-big and ingest-small first.")
        sys.exit(1)

    feature_matrix = np.vstack(all_features)
    logger.info(f"Feature matrix: {feature_matrix.shape}")

    logger.info("Phase 2/4: Training AE1 -> IF -> AE2 ensemble")
    Path(cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
    pipeline = AnomalyPipeline(
        input_dim=feature_matrix.shape[1],
        if_contamination=cfg["if_contamination"],
        anomaly_threshold_percentile=cfg["anomaly_percentile"],
        model_dir=cfg["model_dir"],
    )
    train_stats = pipeline.train(
        feature_matrix,
        ae1_epochs=cfg["ae1_epochs"],
        ae2_epochs=cfg["ae2_epochs"],
        batch_size=cfg["train_batch_size"],
    )

    logger.info("Phase 3/4: Scoring logs and building Knowledge Base")
    km            = KnowledgeManager(persist_dir=cfg["chroma_dir"])
    SCORE_CHUNK   = 50_000
    template_stats: dict = {}
    anomaly_count = 0

    for start in range(0, len(all_metadata), SCORE_CHUNK):
        chunk_features = feature_matrix[start: start + SCORE_CHUNK]
        chunk_meta     = all_metadata[start: start + SCORE_CHUNK]
        scores         = pipeline.score(chunk_features, return_latent=True)

        for i, (is_anom, ae2_err, anom_score, meta) in enumerate(zip(
            scores["is_anomaly"], scores["ae2_errors"],
            scores["anomaly_score"], chunk_meta,
        )):
            tid = meta.get("template_id", "E_UNKNOWN")
            if tid not in template_stats:
                template_stats[tid] = {
                    "template_id": tid, "template_str": meta.get("template_str",""),
                    "frequency": 0, "entropy_sum": 0.0, "burstiness_sum": 0.0,
                    "volatility_sum": 0.0, "time_delta_sum": 0.0, "error_count": 0,
                    "embedding_sum": np.zeros(384, dtype=np.float64),
                }
            ts = template_stats[tid]
            ts["frequency"]      += 1
            ts["entropy_sum"]    += meta.get("entropy",    0.0)
            ts["burstiness_sum"] += meta.get("burstiness", 0.0)
            ts["volatility_sum"] += meta.get("volatility", 0.0)
            ts["time_delta_sum"] += meta.get("time_delta", 0.0)
            if meta.get("level") in ("ERROR","FATAL","CRITICAL"):
                ts["error_count"] += 1
            ts["embedding_sum"] += chunk_features[i, :384].astype(np.float64)

            if is_anom:
                anomaly_count += 1
                emb = chunk_features[i, :384].astype(np.float32)
                emb /= max(np.linalg.norm(emb), 1e-9)
                km.upsert_anomaly(
                    log_id=meta.get("log_id", f"anon_{start+i}"),
                    template_id=tid, raw_line=meta.get("raw_line",""),
                    embedding=emb, ae2_error=float(ae2_err),
                    anomaly_score=float(anom_score),
                    timestamp=meta.get("timestamp",""),
                    service=meta.get("service"),
                )

    template_records = []
    for tid, ts in template_stats.items():
        freq    = ts["frequency"]
        avg_emb = (ts["embedding_sum"] / freq).astype(np.float32)
        avg_emb /= max(np.linalg.norm(avg_emb), 1e-9)
        template_records.append({
            "template_id": tid, "template_str": ts["template_str"],
            "embedding": avg_emb, "frequency": freq,
            "avg_entropy":    ts["entropy_sum"]    / freq,
            "avg_burstiness": ts["burstiness_sum"] / freq,
            "avg_volatility": ts["volatility_sum"] / freq,
            "avg_time_delta": ts["time_delta_sum"] / freq,
            "error_rate":     ts["error_count"]    / freq,
            "is_golden":      (ts["error_count"] / freq) < 0.01,
        })
    km.bulk_upsert_templates(template_records)

    logger.info("Phase 4/4: Running KNN label propagation")
    propagated = km.propagate_labels(
        k_neighbors=cfg["knn_k"],
        confidence_threshold=cfg["label_confidence"],
    )

    elapsed = time.time() - t_start
    _print_section("TRAINING COMPLETE")
    print(f"  Total logs processed   : {total_rows:,}")
    print(f"  Feature matrix shape   : {feature_matrix.shape}")
    print(f"  Normal samples (IF)    : {train_stats['n_normal']:,}")
    print(f"  Point anomalies (IF)   : {train_stats['n_point_anomalies']:,}")
    print(f"  AE2 threshold          : {train_stats['ae2_threshold']:.6f}")
    print(f"  Anomalies in KB        : {anomaly_count:,}")
    print(f"  Templates in KB        : {len(template_stats):,}")
    print(f"  Labels propagated      : {len(propagated):,}")
    print(f"  Total time             : {elapsed:.1f}s")
    print(f"  KB stats               : {km.get_statistics()}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: PROPAGATE
# ══════════════════════════════════════════════════════════════════════════════
def run_propagate(cfg: dict):
    from knowledge_manager import KnowledgeManager
    logger.info("[propagate] Running KNN label propagation")
    km         = KnowledgeManager(persist_dir=cfg["chroma_dir"])
    propagated = km.propagate_labels(
        k_neighbors=cfg["knn_k"],
        confidence_threshold=cfg["label_confidence"],
    )
    _print_section("PROPAGATION COMPLETE")
    print(f"  Labels propagated : {len(propagated):,}")
    print(f"  KB stats          : {km.get_statistics()}")
    return propagated


# ══════════════════════════════════════════════════════════════════════════════
# MODE: ANALYSE  (extended with universal router for --input)
# ══════════════════════════════════════════════════════════════════════════════
def run_analyse(cfg: dict, input_path: str, query: str, stream: bool = True):
    """
    Ingest a new file (any supported format), score it against the trained
    AE2 baseline, retrieve context from ChromaDB, and generate a RAG report.
    """
    from ingestion_engine import LogIngestionEngine
    from feature_extractor import FeatureExtractor
    from anomaly_pipeline import AnomalyPipeline
    from knowledge_manager import KnowledgeManager
    from analyst_rag import AnalystRAG

    logger.info(f"[analyse] Input: {input_path}")

    # Ingest the new file using the universal router
    source_name = None
    if input_path and Path(input_path).exists():
        p   = Path(input_path)
        fmt = _detect_format(str(p))

        if p.is_dir():
            # Directory: ingest all files, then analyse against all new source files
            log_paths = _route_directory(str(p), cfg)
            with LogIngestionEngine(
                db_path=cfg["db_path"],
                config_path=cfg["drain_config"],
                drain_state_path=cfg["drain_state"],
                chunk_size=cfg["chunk_size"],
            ) as engine:
                for lp in log_paths:
                    engine.ingest_file(lp, resume=True)
            # Analyse everything ingested from this directory
            source_name = None   # will analyse all logs
        else:
            # Single file: route to .log then ingest
            log_path    = _route_file_to_log(str(p), cfg)
            source_name = Path(log_path).name
            with LogIngestionEngine(
                db_path=cfg["db_path"],
                config_path=cfg["drain_config"],
                drain_state_path=cfg["drain_state"],
                chunk_size=cfg["chunk_size"],
            ) as engine:
                engine.ingest_file(log_path, resume=True)

    # Load trained pipeline
    pipeline = AnomalyPipeline.load(model_dir=cfg["model_dir"])

    # Extract features
    engine = LogIngestionEngine(
        db_path=cfg["db_path"],
        config_path=cfg["drain_config"],
        drain_state_path=cfg["drain_state"],
        chunk_size=cfg["chunk_size"],
    )
    fe    = FeatureExtractor(batch_embed_size=cfg["embed_batch_size"])
    where = f"source_file = '{source_name}'" if source_name else "1=1"

    all_features, all_metadata = [], []
    for features, metadata in fe.stream_features(
        engine.stream_feature_rows(batch_size=cfg["chunk_size"], where_clause=where)
    ):
        all_features.append(features)
        all_metadata.extend(metadata)
    engine.close()

    if not all_features:
        print("[analyse] No logs found to analyse.")
        return

    feature_matrix = np.vstack(all_features)
    logger.info(f"[analyse] Scoring {feature_matrix.shape[0]:,} entries")

    scores = pipeline.score(feature_matrix, return_latent=True)
    logger.info(f"[analyse] Anomalies detected: {scores['is_anomaly'].sum():,}")

    km      = KnowledgeManager(persist_dir=cfg["chroma_dir"])
    analyst = AnalystRAG(
        knowledge_manager=km,
        ollama_model=cfg["ollama_model"],
        ollama_url=cfg["ollama_url"],
        report_dir=cfg["report_dir"],
    )
    result = analyst.analyse_anomaly_batch(
        anomaly_scores_result=scores,
        feature_metadata=all_metadata,
        embeddings=feature_matrix,
        user_query=query or "Identify root cause and provide remediation steps.",
        stream=stream,
    )

    if not stream:
        _print_section("INCIDENT ANALYSIS REPORT")
        print(result["report"])
        _print_section("EXECUTIVE SUMMARY")
        print(result["executive_summary"])

    print(f"\n[analyse] Done. Reports saved to: {cfg['report_dir']}/")
    print(f"  Anomalies : {result['n_anomalies']}")
    print(f"  Services  : {result.get('affected_services', [])}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: CHAT
# ══════════════════════════════════════════════════════════════════════════════
def run_chat(cfg: dict):
    from knowledge_manager import KnowledgeManager
    from analyst_rag import AnalystRAG
    km = KnowledgeManager(persist_dir=cfg["chroma_dir"])
    analyst = AnalystRAG(
        knowledge_manager=km,
        ollama_model=cfg["ollama_model"],
        ollama_url=cfg["ollama_url"],
        report_dir=cfg["report_dir"],
    )
    analyst.interactive_chat()


# ══════════════════════════════════════════════════════════════════════════════
# MODE: STATUS
# ══════════════════════════════════════════════════════════════════════════════
def run_status(cfg: dict):
    import duckdb, requests
    _print_section("AIOps System Status")

    try:
        con = duckdb.connect(cfg["db_path"], read_only=True)
        row = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT template_id) FROM raw_logs"
        ).fetchone()
        con.close()
        print(f"  DuckDB logs          : {row[0]:,}  [OK]")
        print(f"  Unique templates     : {row[1]:,}")
    except Exception as e:
        print(f"  DuckDB               : [MISSING] {e}")

    model_dir = Path(cfg["model_dir"])
    for name, fname in [("AE1","ae1.pt"),("AE2","ae2.pt"),("IF","isolation_forest.pkl")]:
        ok = (model_dir / fname).exists()
        print(f"  Model {name:<6}         : {'[OK]' if ok else '[NOT TRAINED]'}")

    try:
        from knowledge_manager import KnowledgeManager
        stats = KnowledgeManager(persist_dir=cfg["chroma_dir"]).get_statistics()
        print(f"  ChromaDB templates   : {stats['templates']:,}  [OK]")
        print(f"  ChromaDB anomalies   : {stats['anomalies']:,}")
        print(f"  Labelled templates   : {stats['labelled_templates']:,}")
    except Exception as e:
        print(f"  ChromaDB             : [MISSING] {e}")

    try:
        resp   = requests.get(f"{cfg['ollama_url']}/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"  Ollama               : [OK] ({', '.join(models[:3])})")
    except Exception:
        print(f"  Ollama               : [NOT RUNNING] start with: ollama serve")

    reports = list(Path(cfg["report_dir"]).glob("*.json")) \
              if Path(cfg["report_dir"]).exists() else []
    print(f"  Reports generated    : {len(reports)}")

    # Show supported formats
    print()
    print("  Supported input formats:")
    for ext, fmt in sorted(_FORMAT_MAP.items()):
        print(f"    {ext:<8} -> {fmt}")

    _print_section("END STATUS")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: ALL
# ══════════════════════════════════════════════════════════════════════════════
def run_all(cfg: dict, big_csv: str, small_csv: str):
    logger.info("=== FULL PIPELINE - START ===")
    t0 = time.time()

    run_ingest_big(cfg, big_csv)
    run_ingest_small(cfg, small_csv)
    run_train(cfg)
    run_analyse(
        cfg, input_path=big_csv,
        query=(
            "Which services have the highest anomaly rates? "
            "What are the dominant error patterns and what corrective actions are needed?"
        ),
        stream=True,
    )

    elapsed = time.time() - t0
    logger.info(f"=== FULL PIPELINE - DONE in {elapsed:.1f}s ===")
    _print_section("FULL PIPELINE COMPLETE")
    print(f"  Total time    : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Reports saved : {cfg['report_dir']}/")
    print()
    print("  Next steps:")
    print("    python main.py --mode status")
    print("    python main.py --mode chat")
    print("    python main.py --mode analyse --input <any_file> --query \"<question>\"")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _print_section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="AIOps Log Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["all","ingest-big","ingest-small","ingest","train",
                 "propagate","analyse","chat","status"],
        help="Operation mode (use 'all' to run everything from scratch)",
    )
    parser.add_argument("--big",      help="Apache Log CSV (no header). Required for: all, ingest-big")
    parser.add_argument("--small",    help="Structured anomaly CSV. Required for: all, ingest-small")
    parser.add_argument("--input",    help="Any supported file/directory for ingest or analyse")
    parser.add_argument("--config",   default="config.json", help="JSON config file path")
    parser.add_argument("--query",    default="",            help="RAG analysis question")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for train (debug)")
    parser.add_argument("--no-stream",action="store_true",   help="Disable LLM streaming")
    parser.add_argument("--model",    help="Override Ollama model (e.g. llama3:8b)")

    args = parser.parse_args()
    cfg  = load_config(args.config)
    if args.model:
        cfg["ollama_model"] = args.model

    if args.mode == "all":
        _require(parser, args, big="--big", small="--small")
        run_all(cfg, args.big, args.small)
    elif args.mode == "ingest-big":
        _require(parser, args, big="--big")
        run_ingest_big(cfg, args.big)
    elif args.mode == "ingest-small":
        _require(parser, args, small="--small")
        run_ingest_small(cfg, args.small)
    elif args.mode == "ingest":
        _require(parser, args, input="--input")
        run_ingest(cfg, args.input)
    elif args.mode == "train":
        run_train(cfg, max_rows=args.max_rows)
    elif args.mode == "propagate":
        run_propagate(cfg)
    elif args.mode == "analyse":
        run_analyse(cfg, args.input, args.query, stream=not args.no_stream)
    elif args.mode == "chat":
        run_chat(cfg)
    elif args.mode == "status":
        run_status(cfg)


def _require(parser, args, **kwargs):
    for flag, name in kwargs.items():
        val = getattr(args, flag.replace("-", "_"), None)
        if not val:
            parser.error(f"{name} is required for --mode {args.mode}")
        if not Path(val).exists():
            parser.error(f"{name} file not found: {val}")


if __name__ == "__main__":
    main()
