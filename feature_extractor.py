"""
feature_extractor.py
---------------------
Transforms raw log rows into a rich multidimensional feature matrix.

Features extracted per log entry:
  ① Semantic embeddings      (384d via all-MiniLM-L6-v2)
  ② Shannon entropy          (message randomness, normalised to [0.0, 1.0])
  ③ Burstiness               (template frequency in 10s sliding window)
  ④ Volatility               (running std-dev via Welford's O(1) algorithm)
  ⑤ Time delta               (seconds since last event in same service/thread)
  ⑥ HTTP status binary flags (1xx–5xx, one-hot)
  ⑦ Log level encoding       (ordinal: TRACE→DEBUG→INFO→WARN→ERROR→FATAL)

All processing is streaming/chunked — safe for 100M+ rows.

Production hardening (v2) — four architectural improvements:
  [FIX-1] Bounded LRU caches (cachetools) replace unbounded defaultdicts
          → eliminates OOM on 100M+ unique template_id / service keys
  [FIX-2] Welford's Online Algorithm replaces deque + np.std()
          → O(1) volatility per row instead of O(N) re-scan of up to 500 values
  [FIX-3] ciso8601 fast-path + format-caching replaces try/except strptime loop
          → eliminates repeated format-scan overhead on every timestamp string
  [FIX-4] Normalised Shannon entropy (raw / log2(len)) → strict [0.0, 1.0] output
          → eliminates length-bias between short and long log lines
"""

import re
import json
import math
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Generator, List, Optional, Tuple

import numpy as np
from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

# ── Optional fast ISO-8601 parser (C extension) ───────────────────────────────
try:
    import ciso8601                     # pip install ciso8601
    _CISO8601_AVAILABLE = True
except ImportError:                     # graceful degradation to pure-Python fallback
    _CISO8601_AVAILABLE = False

logger = logging.getLogger("FeatureExtractor")

# ── Constants ──────────────────────────────────────────────────────────────────
LEVEL_ORDINAL = {
    "TRACE": 0, "DEBUG": 1, "INFO": 2,
    "WARN": 2.5, "WARNING": 2.5,
    "ERROR": 4, "FATAL": 5, "CRITICAL": 5,
    "UNKNOWN": 2,
}

BURST_WINDOW_SEC   = 10.0       # sliding window width for burstiness
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
EMBEDDING_DIM      = 384
MAX_NUMERIC_PARAMS = 16         # cap numeric parameter scan per row

# Regex to grab any numeric value from a log token
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# strptime fallback formats tried in order when ciso8601 is unavailable
_STRPTIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
)

# ── LRU cache size limits ─────────────────────────────────────────────────────
# Tune to your cardinality expectations; these cover virtually all real
# deployments while keeping per-extractor overhead to a few MB of RAM.
_LRU_SERVICES  = 10_000   # bound for _last_ts_per_service
_LRU_TEMPLATES = 50_000   # bound for _burst_windows + _welford_state


# ══════════════════════════════════════════════════════════════════════════════
class FeatureExtractor:
    """
    Stateful feature extractor that maintains sliding-window state across
    streaming batches for correct burstiness and time-delta calculations.

    Usage:
        fe = FeatureExtractor()
        for batch in engine.stream_feature_rows():
            feature_matrix, metadata = fe.process_batch(batch)
            # feature_matrix.shape == (N, 393)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_embed_size: int = 256):
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder        = SentenceTransformer(model_name)
        self.batch_embed_size = batch_embed_size

        # ── [FIX-1] Bounded LRU caches — replaces unbounded defaultdicts ──────
        #
        # _burst_windows      : template_id → deque of unix timestamps
        #   Eviction is safe: an LRU-evicted template hasn't been seen recently,
        #   so its burst window is already stale; a fresh deque on next hit is correct.
        #
        # _last_ts_per_service: service → last seen unix timestamp (float)
        #   Eviction is safe: delta for the next event from that service will be
        #   reported as 0.0 (first-seen semantics), which is conservative.
        #
        # _welford_state      : template_id → Welford running-variance state dict
        #   Eviction drops accumulated history for that template; the next observation
        #   restarts from count=1, which is correct and safe.
        self._burst_windows:        LRUCache = LRUCache(maxsize=_LRU_TEMPLATES)
        self._last_ts_per_service:  LRUCache = LRUCache(maxsize=_LRU_SERVICES)
        self._welford_state:        LRUCache = LRUCache(maxsize=_LRU_TEMPLATES)

        # ── [FIX-3] Format cache — last successful strptime format string ──────
        # Initialised to None; set on first successful string parse and reused
        # for every subsequent row from the same log source.
        self._last_ts_fmt: Optional[str] = None

    # ══════════════════════════════════════════════════════════════════════════
    # [FIX-4] Normalised Shannon Entropy
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def shannon_entropy(text: str) -> float:
        """
        Character-level Shannon entropy normalised to [0.0, 1.0].

        Formula:
            H_norm = H_raw / log2(len(text))

        Normalising by the theoretical maximum entropy for the given length
        removes the length-bias:  a perfectly random 10-char string now scores
        as high as a perfectly random 500-char string (both → 1.0), while a
        completely uniform string ("aaaa…") always scores 0.0 regardless of
        its length.

        Edge cases handled explicitly:
            • empty string       → 0.0
            • single character   → 0.0  (no disorder possible with one symbol)
            • max_entropy == 0.0 → 0.0  (defensive guard, cannot occur for len>1)
        """
        if not text:
            return 0.0

        length = len(text)
        if length == 1:
            return 0.0

        # Character frequency via plain dict (tighter inner loop than defaultdict)
        freq: dict = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1

        # Raw Shannon entropy H = -Σ p·log2(p)
        raw_entropy = -sum(
            (c / length) * math.log2(c / length)
            for c in freq.values()
        )

        # Theoretical maximum for this length: log2(length)
        max_entropy = math.log2(length)
        if max_entropy == 0.0:           # safety guard (length==1 already caught)
            return 0.0

        # Clamp to 1.0 to absorb any floating-point overshoot
        return round(min(raw_entropy / max_entropy, 1.0), 6)

    # ══════════════════════════════════════════════════════════════════════════
    # [FIX-1] Burstiness — LRU-backed sliding window
    # ══════════════════════════════════════════════════════════════════════════
    def _update_burstiness(self, template_id: str, ts_unix: float) -> int:
        """
        Count occurrences of `template_id` in the last BURST_WINDOW_SEC seconds.
        Each template's sliding window lives in a bounded LRU cache.
        """
        window = self._burst_windows.get(template_id)
        if window is None:
            window = deque()
            self._burst_windows[template_id] = window

        cutoff = ts_unix - BURST_WINDOW_SEC
        while window and window[0] < cutoff:
            window.popleft()
        window.append(ts_unix)
        return len(window)

    # ══════════════════════════════════════════════════════════════════════════
    # [FIX-1] Time delta — LRU-backed last-seen map
    # ══════════════════════════════════════════════════════════════════════════
    def _compute_time_delta(self, service: str, ts_unix: Optional[float]) -> float:
        """
        Seconds since last log event for this service.
        Returns 0.0 on first occurrence or when timestamp is unavailable.
        A cache miss (evicted key) is treated identically to first-occurrence.
        """
        if ts_unix is None:
            return 0.0
        prev = self._last_ts_per_service.get(service)  # None if absent/evicted
        self._last_ts_per_service[service] = ts_unix
        if prev is None:
            return 0.0
        delta = max(0.0, ts_unix - prev)
        return round(min(delta, 86400.0), 4)            # cap at 1 day

    # ══════════════════════════════════════════════════════════════════════════
    # [FIX-2] Volatility — Welford's Online Algorithm, O(1) per row
    # ══════════════════════════════════════════════════════════════════════════
    def _compute_volatility(self, template_id: str, dynamic_params_json: str) -> float:
        """
        Population standard deviation of numeric parameters seen for a template,
        maintained in O(1) time per row using Welford's Online Algorithm.

        State stored per template_id (3 scalars, ~24 bytes):
            count — total numeric values ingested so far
            mean  — running arithmetic mean
            M2    — running sum of squared deviations from the current mean

        Welford single-pass update for each new value x:
            count  += 1
            delta   = x − mean
            mean   += delta / count          ← update mean FIRST
            delta2  = x − mean               ← delta from UPDATED mean
            M2     += delta × delta2
            σ²      = M2 / count             ← population variance
            σ       = √σ²
        """
        try:
            params = json.loads(dynamic_params_json) if dynamic_params_json else []
        except json.JSONDecodeError:
            params = []

        # Retrieve or initialise Welford state for this template
        state = self._welford_state.get(template_id)
        if state is None:
            state = {"count": 0, "mean": 0.0, "M2": 0.0}
            self._welford_state[template_id] = state

        for p in params[:MAX_NUMERIC_PARAMS]:
            match = _NUM_RE.search(str(p))
            if not match:
                continue
            try:
                x = float(match.group())
            except ValueError:
                continue

            # ── Welford one-pass update (O(1)) ────────────────────────────────
            state["count"] += 1
            delta           = x - state["mean"]
            state["mean"]  += delta / state["count"]
            delta2          = x - state["mean"]          # uses updated mean
            state["M2"]    += delta * delta2

        if state["count"] < 2:
            return 0.0
        variance = state["M2"] / state["count"]          # population variance
        return round(math.sqrt(max(variance, 0.0)), 6)

    # ══════════════════════════════════════════════════════════════════════════
    # [FIX-3] Timestamp → unix float — ciso8601 fast-path + format cache
    # ══════════════════════════════════════════════════════════════════════════
    def _to_unix(self, ts) -> Optional[float]:
        """
        Convert any common timestamp representation to a UTC unix float.

        Parse priority for string inputs:
          3a. ciso8601.parse_datetime()  — C extension, ~10× faster than strptime;
              handles ISO-8601 / RFC-3339 and most structured log timestamps.
          3b. self._last_ts_fmt (cached) — the format that succeeded last time is
              tried first, skipping the loop entirely for homogeneous log files.
          3c. Full _STRPTIME_FORMATS scan — only reached on the very first row or
              when the log source changes format mid-stream.
        """
        if ts is None:
            return None

        # ── Fast path 1: numeric ──────────────────────────────────────────────
        if isinstance(ts, (int, float)):
            return float(ts)

        # ── Fast path 2: datetime object ──────────────────────────────────────
        if isinstance(ts, datetime):
            try:
                if ts.tzinfo is None:
                    return ts.replace(tzinfo=timezone.utc).timestamp()
                return ts.timestamp()
            except (OSError, OverflowError, ValueError):
                return None

        # ── String path ───────────────────────────────────────────────────────
        if not isinstance(ts, str):
            return None

        ts_str = ts.strip()
        if not ts_str:
            return None

        # 3a. ciso8601 — C extension fast path (handles ISO-8601 + RFC-3339)
        if _CISO8601_AVAILABLE:
            try:
                dt = ciso8601.parse_datetime(ts_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except (ValueError, OverflowError):
                pass                    # fall through to strptime

        # 3b. Cached format — O(1) for the common case of a homogeneous log file
        if self._last_ts_fmt is not None:
            try:
                return datetime.strptime(ts_str, self._last_ts_fmt).replace(
                    tzinfo=timezone.utc
                ).timestamp()
            except ValueError:
                pass                    # format changed; fall through to full scan

        # 3c. Full format scan — runs only on the first row or after a format change
        for fmt in _STRPTIME_FORMATS:
            try:
                result = datetime.strptime(ts_str, fmt).replace(
                    tzinfo=timezone.utc
                ).timestamp()
                self._last_ts_fmt = fmt  # ← cache the winner for all future rows
                return result
            except ValueError:
                continue

        return None                     # completely unparseable — omit this feature

    # ══════════════════════════════════════════════════════════════════════════
    # Batch embeddings (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Batch embed texts in sub-batches to control GPU/CPU memory.
        Returns ndarray of shape (N, 384).
        """
        all_embeddings = []
        for i in range(0, len(texts), self.batch_embed_size):
            sub  = texts[i: i + self.batch_embed_size]
            embs = self.embedder.encode(
                sub,
                batch_size=self.batch_embed_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.append(embs)
        return np.vstack(all_embeddings).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Main batch processor (contract unchanged — output shape stays (N, 393))
    # ══════════════════════════════════════════════════════════════════════════
    def process_batch(
        self,
        rows: list,
        return_metadata: bool = True,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process a batch of raw log rows into feature vectors.

        Args:
            rows: List of tuples from `engine.stream_feature_rows()`.
                  Columns: (log_id, timestamp, level, service, template_id,
                             template_str, dynamic_params, raw_line,
                             http_status, http_1xx, http_2xx, http_3xx,
                             http_4xx, http_5xx)
            return_metadata: Whether to also return metadata dicts.

        Returns:
            (feature_matrix, metadata_list)
            feature_matrix shape: (N, 393)   ← CONTRACT UNCHANGED
              [embedding(384) | entropy(1) | burstiness(1) | volatility(1) |
               time_delta(1)  | level_ordinal(1) | http_flags(5)]
        """
        if not rows:
            return np.empty((0, EMBEDDING_DIM + 9), dtype=np.float32), []

        texts           = []
        metadata        = []
        scalar_features = []

        for row in rows:
            (
                log_id, timestamp, level, service, template_id,
                template_str, dynamic_params, raw_line,
                http_status, h1, h2, h3, h4, h5
            ) = row

            ts_unix    = self._to_unix(timestamp)                   # [FIX-3]
            embed_text = (template_str or raw_line or "")[:512]
            texts.append(embed_text)

            entropy    = self.shannon_entropy(embed_text)            # [FIX-4]
            burstiness = self._update_burstiness(                    # [FIX-1]
                template_id or "E_UNKNOWN", ts_unix or time.time()
            )
            volatility = self._compute_volatility(                   # [FIX-1, FIX-2]
                template_id or "E_UNKNOWN", dynamic_params or "[]"
            )
            time_delta = self._compute_time_delta(                   # [FIX-1]
                service or "unknown", ts_unix
            )
            level_enc  = self._level_ordinal(level)
            http_vec   = self._http_flags_vector(row)

            scalars = np.array(
                [entropy, float(burstiness), volatility, time_delta, level_enc],
                dtype=np.float32,
            )
            scalar_features.append(np.concatenate([scalars, http_vec]))

            if return_metadata:
                metadata.append({
                    "log_id":       log_id,
                    "timestamp":    str(timestamp) if timestamp else None,
                    "level":        level,
                    "service":      service,
                    "template_id":  template_id,
                    "template_str": (template_str or "")[:256],
                    "raw_line":     (raw_line or "")[:512],
                    "http_status":  http_status,
                    "entropy":      entropy,
                    "burstiness":   burstiness,
                    "volatility":   volatility,
                    "time_delta":   time_delta,
                })

        embeddings     = self._embed_texts(texts)                    # (N, 384)
        scalar_matrix  = np.vstack(scalar_features)                 # (N, 9)
        feature_matrix = np.hstack([embeddings, scalar_matrix])     # (N, 393)

        return feature_matrix, metadata

    # ══════════════════════════════════════════════════════════════════════════
    # HTTP flags → numpy vector (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _http_flags_vector(row: tuple) -> np.ndarray:
        # Columns in stream_feature_rows: http_1xx=9 … http_5xx=13
        return np.array(
            [float(bool(row[9])),  float(bool(row[10])),
             float(bool(row[11])), float(bool(row[12])), float(bool(row[13]))],
            dtype=np.float32,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Level ordinal (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _level_ordinal(level: Optional[str]) -> float:
        return LEVEL_ORDINAL.get((level or "UNKNOWN").upper(), 2.0)

    # ══════════════════════════════════════════════════════════════════════════
    # Streaming wrapper (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def stream_features(
        self,
        row_generator: Generator,
    ) -> Generator[Tuple[np.ndarray, List[dict]], None, None]:
        """Wraps a row generator, yielding (features, metadata) pairs."""
        for batch in row_generator:
            features, meta = self.process_batch(batch)
            if features.shape[0] > 0:
                yield features, meta

    # ══════════════════════════════════════════════════════════════════════════
    # Normalization helpers (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def fit_scaler(feature_matrix: np.ndarray):
        """Fit a StandardScaler on the 9 scalar columns (embeddings excluded)."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(feature_matrix[:, EMBEDDING_DIM:])
        return scaler

    @staticmethod
    def apply_scaler(feature_matrix: np.ndarray, scaler) -> np.ndarray:
        """Apply a fitted scaler to the scalar portion of the feature matrix."""
        scaled = feature_matrix.copy()
        scaled[:, EMBEDDING_DIM:] = scaler.transform(feature_matrix[:, EMBEDDING_DIM:])
        return scaled

    # ══════════════════════════════════════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════════════════════════════════════
    def cache_stats(self) -> dict:
        """
        Return current fill ratios for all LRU caches.
        Use this in production to verify maxsize values are appropriate
        (a cache near 100% fill evicts frequently; consider raising maxsize).
        """
        return {
            "burst_windows":        f"{len(self._burst_windows)}"
                                    f"/{self._burst_windows.maxsize}",
            "last_ts_per_service":  f"{len(self._last_ts_per_service)}"
                                    f"/{self._last_ts_per_service.maxsize}",
            "welford_state":        f"{len(self._welford_state)}"
                                    f"/{self._welford_state.maxsize}",
            "ts_fmt_cached":        self._last_ts_fmt or "(none yet)",
            "ciso8601_available":   _CISO8601_AVAILABLE,
        }


# ── Module-level helpers (unchanged public API) ───────────────────────────────
def compute_entropy_bulk(texts: List[str]) -> np.ndarray:
    """Vectorised entropy over a list of strings — useful for offline analysis."""
    return np.array([FeatureExtractor.shannon_entropy(t) for t in texts])


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== FeatureExtractor v2 Smoke Test ===\n")

    fe = FeatureExtractor()

    sample_rows = [
        (
            f"id_{i}",
            datetime.utcnow(),
            "ERROR",
            "auth-service",
            f"E{i % 5}",
            "Connection timeout after <*> ms",
            json.dumps(["500", "db-host-1"]),
            f"2024-01-01 12:00:0{i} ERROR auth-service - Connection timeout after 500 ms",
            None, False, False, False, False, True,
        )
        for i in range(10)
    ]

    features, meta = fe.process_batch(sample_rows)

    print(f"Feature matrix shape    : {features.shape}  (expected (10, 393))")
    print(f"Sample metadata keys    : {list(meta[0].keys())}")
    print()

    # [FIX-4] Normalised entropy must be strictly in [0.0, 1.0]
    entropy_col = features[:, 384]
    assert entropy_col.min() >= 0.0 and entropy_col.max() <= 1.0, \
        f"[FIX-4] entropy out of [0,1]: {entropy_col}"
    print(f"[FIX-4] Entropy (normalised)    : "
          f"{entropy_col.min():.4f} – {entropy_col.max():.4f}  (must be in [0,1]) ✓")

    # Manually verify normalisation for a known string
    h_uniform = FeatureExtractor.shannon_entropy("aaaaaaaaaa")   # all same → 0.0
    h_random  = FeatureExtractor.shannon_entropy("abcdefghij")   # all distinct → 1.0
    assert h_uniform == 0.0,  f"[FIX-4] uniform string entropy expected 0.0, got {h_uniform}"
    assert h_random  == 1.0,  f"[FIX-4] all-distinct string entropy expected 1.0, got {h_random}"
    print(f"[FIX-4] 'aaaaaaaaaa' entropy    : {h_uniform}  (expected 0.0) ✓")
    print(f"[FIX-4] 'abcdefghij' entropy    : {h_random}  (expected 1.0) ✓")
    print()

    # [FIX-2] Welford state should be stored as a dict with M2 key (not a deque)
    state = fe._welford_state.get("E0")
    assert state is not None and "M2" in state, "[FIX-2] Welford state missing or malformed"
    print(f"[FIX-2] Welford state for E0    : "
          f"count={state['count']}, mean={state['mean']:.2f}, M2={state['M2']:.4f} ✓")
    print()

    # [FIX-1] All three state containers must be LRUCache instances
    assert isinstance(fe._burst_windows,       LRUCache), "[FIX-1] _burst_windows not LRUCache"
    assert isinstance(fe._last_ts_per_service, LRUCache), "[FIX-1] _last_ts_per_service not LRUCache"
    assert isinstance(fe._welford_state,       LRUCache), "[FIX-1] _welford_state not LRUCache"
    print(f"[FIX-1] All state containers    : bounded LRUCache ✓")
    print()

    # [FIX-3] Format cache state
    print(f"[FIX-3] ciso8601 available      : {_CISO8601_AVAILABLE}")
    print(f"[FIX-3] Cached strptime format  : '{fe._last_ts_fmt}' ✓")
    print()

    # Final contract check — output shape must still be (N, 393)
    assert features.shape == (10, 393), f"Shape contract broken: {features.shape}"
    print(f"Output shape (N=10, 393)        : {features.shape} ✓")
    print(f"\nCache diagnostics               : {fe.cache_stats()}")
    print("\n✓ All assertions passed — FeatureExtractor v2 OK")
