"""
knowledge_manager.py  (v4 — final production build)
-----------------------------------------------------
ChromaDB-backed knowledge base:
  * CRUD for log templates + golden baselines
  * Label propagation via KNN
  * Hybrid retrieval: semantic cosine + BM25 keyword

Cumulative fixes:
  [FIX-A] "ids" removed from every include= list.
          ChromaDB >= 0.5 always returns ids automatically.

  [FIX-B] Paginated fetches (_PAGE_SIZE=2000) everywhere.
          col.get() with no limit crashes SQLite at 100K+ rows.

  [FIX-C] BM25 rebuilt lazily (dirty flag) not on every batch upsert.

  [FIX-D] _safe_list() helper replaces every `value or []` pattern.
          When ChromaDB returns embeddings as a numpy array (newer builds),
          `numpy_array or []` raises "ambiguous truth value" ValueError.
          _safe_list() uses an explicit `is None` check instead.
"""

import json
import logging
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import chromadb
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger("KnowledgeManager")

COL_TEMPLATES = "log_templates"
COL_ANOMALIES = "anomaly_events"
COL_BASELINES = "normal_baselines"

# Max rows per ChromaDB get() page — well below SQLite's 32,766-variable limit
_PAGE_SIZE = 2_000


# ══════════════════════════════════════════════════════════════════════════════
# [FIX-D]  Safe result accessor
# ══════════════════════════════════════════════════════════════════════════════
def _safe_list(value) -> list:
    """
    Convert any ChromaDB result field to a plain Python list safely.

    Handles every type ChromaDB may return for ids / documents / metadatas /
    embeddings / distances:
      None               -> []          (missing field)
      []                 -> []          (empty page)
      [[0.1, ...], ...]  -> as-is       (list of vectors / strings)
      np.ndarray         -> .tolist()   (newer chromadb builds return arrays)

    The critical point: `numpy_array or []` raises ValueError because numpy
    overloads __bool__ to raise for arrays with more than one element.
    `value is None` is the only boolean check that is safe for all types.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _safe_nested_list(value) -> list:
    """
    For query() results where the field is [[...]] (one list per query).
    Returns the inner list for the first (and only) query embedding.
    """
    outer = _safe_list(value)
    if not outer:
        return []
    inner = outer[0]
    if isinstance(inner, np.ndarray):
        return inner.tolist()
    return list(inner) if inner is not None else []


# ══════════════════════════════════════════════════════════════════════════════
# BM25
# ══════════════════════════════════════════════════════════════════════════════
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self._corpus_tokens: List[List[str]] = []
        self._doc_ids:       List[str]       = []
        self._idf:           Dict[str, float] = {}
        self._avg_dl:        float            = 0.0

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, texts: List[str], doc_ids: List[str]):
        import math
        self._corpus_tokens = [self._tokenise(t) for t in texts]
        self._doc_ids       = doc_ids
        N  = len(texts)
        df: Counter = Counter()
        for tokens in self._corpus_tokens:
            df.update(set(tokens))
        self._idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        self._avg_dl = sum(len(t) for t in self._corpus_tokens) / max(N, 1)

    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q_tokens = self._tokenise(query)
        if not q_tokens:
            return []
        scores = []
        for doc_tokens, doc_id in zip(self._corpus_tokens, self._doc_ids):
            dl     = len(doc_tokens)
            tf_map = Counter(doc_tokens)
            score  = 0.0
            for term in q_tokens:
                tf  = tf_map.get(term, 0)
                idf = self._idf.get(term, 0.0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                score += idf * num / max(den, 1e-9)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# KnowledgeManager
# ══════════════════════════════════════════════════════════════════════════════
class KnowledgeManager:

    def __init__(
        self,
        persist_dir:     str = "chromadb_store",
        bm25_cache_path: str = "bm25_cache.pkl",
    ):
        self.persist_dir     = str(Path(persist_dir).resolve())
        self.bm25_cache_path = bm25_cache_path

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        logger.info(f"ChromaDB initialised -> {self.persist_dir}")

        self.col_templates = self.client.get_or_create_collection(
            name=COL_TEMPLATES, metadata={"hnsw:space": "cosine"},
        )
        self.col_anomalies = self.client.get_or_create_collection(
            name=COL_ANOMALIES, metadata={"hnsw:space": "cosine"},
        )
        self.col_baselines = self.client.get_or_create_collection(
            name=COL_BASELINES, metadata={"hnsw:space": "cosine"},
        )

        self._bm25_dirty: bool          = False
        self.bm25:        Optional[BM25] = None
        self._load_or_build_bm25()

    # ── BM25 ──────────────────────────────────────────────────────────────────
    def _load_or_build_bm25(self):
        if Path(self.bm25_cache_path).exists():
            try:
                with open(self.bm25_cache_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                logger.info("BM25 index loaded from cache")
                return
            except Exception:
                pass
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """[FIX-B, FIX-D] Paginated, numpy-safe BM25 rebuild."""
        all_ids:  List[str] = []
        all_docs: List[str] = []
        offset = 0

        while True:
            try:
                batch = self.col_templates.get(
                    include=["documents"],
                    limit=_PAGE_SIZE,
                    offset=offset,
                )
            except Exception as e:
                logger.warning(f"BM25 rebuild page failed (offset={offset}): {e}")
                break

            # [FIX-D] use _safe_list — never `or []`
            batch_ids  = _safe_list(batch.get("ids"))
            batch_docs = _safe_list(batch.get("documents"))
            if not batch_ids:
                break

            all_ids.extend(batch_ids)
            all_docs.extend(batch_docs)
            if len(batch_ids) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

        if all_ids:
            self.bm25 = BM25()
            self.bm25.fit(all_docs, all_ids)
            try:
                with open(self.bm25_cache_path, "wb") as f:
                    pickle.dump(self.bm25, f)
            except Exception as e:
                logger.warning(f"BM25 cache write failed: {e}")
            logger.info(f"BM25 index built with {len(all_ids)} templates")
        else:
            logger.info("No templates yet - BM25 will be built on first upsert")

    def _ensure_bm25_fresh(self):
        """[FIX-C] Lazy rebuild — only when dirty."""
        if self._bm25_dirty:
            self._rebuild_bm25()
            self._bm25_dirty = False

    # ── Template upsert ───────────────────────────────────────────────────────
    def upsert_template(
        self,
        template_id:      str,
        template_str:     str,
        embedding:        np.ndarray,
        baseline_metrics: Dict,
        label:            Optional[str] = None,
    ):
        self.col_templates.upsert(
            ids=[template_id],
            embeddings=[embedding.tolist()],
            documents=[template_str],
            metadatas=[{
                "template_id":    template_id,
                "label":          label or "unlabelled",
                "frequency":      int(baseline_metrics.get("frequency",      0)),
                "avg_entropy":    float(baseline_metrics.get("avg_entropy",    0.0)),
                "avg_burstiness": float(baseline_metrics.get("avg_burstiness", 0.0)),
                "avg_volatility": float(baseline_metrics.get("avg_volatility", 0.0)),
                "avg_time_delta": float(baseline_metrics.get("avg_time_delta", 0.0)),
                "error_rate":     float(baseline_metrics.get("error_rate",     0.0)),
                "is_golden":      bool(baseline_metrics.get("is_golden",       False)),
                "metrics_json":   json.dumps(baseline_metrics),
            }],
        )
        self._bm25_dirty = True

    def bulk_upsert_templates(self, records: List[Dict]):
        """[FIX-C] Marks dirty once after all batches — no per-batch BM25 rebuild."""
        if not records:
            return
        ids        = [r["template_id"] for r in records]
        embeddings = [r["embedding"].tolist() for r in records]
        documents  = [r["template_str"] for r in records]
        metadatas  = [
            {
                "template_id":    r["template_id"],
                "label":          r.get("label") or "unlabelled",
                "frequency":      int(r.get("frequency",      0)),
                "avg_entropy":    float(r.get("avg_entropy",    0.0)),
                "avg_burstiness": float(r.get("avg_burstiness", 0.0)),
                "avg_volatility": float(r.get("avg_volatility", 0.0)),
                "avg_time_delta": float(r.get("avg_time_delta", 0.0)),
                "error_rate":     float(r.get("error_rate",     0.0)),
                "is_golden":      bool(r.get("is_golden",       False)),
                "metrics_json":   json.dumps(
                    {k: v for k, v in r.items()
                     if k not in ("embedding", "template_str")}
                ),
            }
            for r in records
        ]
        for i in range(0, len(ids), _PAGE_SIZE):
            self.col_templates.upsert(
                ids=ids[i: i + _PAGE_SIZE],
                embeddings=embeddings[i: i + _PAGE_SIZE],
                documents=documents[i: i + _PAGE_SIZE],
                metadatas=metadatas[i: i + _PAGE_SIZE],
            )
        self._bm25_dirty = True
        logger.info(f"Bulk upserted {len(records)} templates")

    # ── Anomaly storage ───────────────────────────────────────────────────────
    def upsert_anomaly(
        self,
        log_id:        str,
        template_id:   str,
        raw_line:      str,
        embedding:     np.ndarray,
        ae2_error:     float,
        anomaly_score: float,
        timestamp:     str,
        label:         Optional[str] = None,
        service:       Optional[str] = None,
    ):
        self.col_anomalies.upsert(
            ids=[log_id],
            embeddings=[embedding.tolist()],
            documents=[raw_line[:512]],
            metadatas=[{
                "template_id":   template_id,
                "ae2_error":     float(ae2_error),
                "anomaly_score": float(anomaly_score),
                "timestamp":     timestamp or "",
                "label":         label or "unlabelled",
                "service":       service or "unknown",
                "raw_line":      raw_line[:512],
            }],
        )

    # ── Normal baseline storage ───────────────────────────────────────────────
    def store_normal_baseline(self, template_id: str, embedding: np.ndarray, metrics: Dict):
        self.col_baselines.upsert(
            ids=[template_id],
            embeddings=[embedding.tolist()],
            documents=[template_id],
            metadatas=[{"metrics_json": json.dumps(metrics)}],
        )

    # ── Label propagation ─────────────────────────────────────────────────────
    def propagate_labels(
        self,
        k_neighbors:          int   = 5,
        confidence_threshold: float = 0.6,
    ) -> Dict[str, str]:
        """
        KNN label propagation from labelled to unlabelled templates.
        [FIX-B] Paginated.  [FIX-D] _safe_list() everywhere.
        """
        all_ids:        List[str]  = []
        all_embeddings: List       = []
        all_metadatas:  List[Dict] = []
        offset = 0

        while True:
            try:
                batch = self.col_templates.get(
                    include=["embeddings", "metadatas"],
                    limit=_PAGE_SIZE,
                    offset=offset,
                )
            except Exception as e:
                logger.warning(f"propagate_labels page fetch failed: {e}")
                break

            # [FIX-D] _safe_list handles None AND numpy arrays
            batch_ids   = _safe_list(batch.get("ids"))
            batch_embs  = _safe_list(batch.get("embeddings"))
            batch_metas = _safe_list(batch.get("metadatas"))

            if not batch_ids:
                break

            all_ids.extend(batch_ids)
            all_embeddings.extend(batch_embs)
            all_metadatas.extend(batch_metas)

            if len(batch_ids) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

        if not all_ids:
            logger.warning("No templates in KB for label propagation")
            return {}

        # Convert embedding list to numpy array safely
        # Each element might be a list or a numpy array
        try:
            embeddings = np.array(
                [np.array(e, dtype=np.float32) for e in all_embeddings],
                dtype=np.float32,
            )
        except Exception as e:
            logger.error(f"Could not convert embeddings to numpy array: {e}")
            return {}

        labelled_idx   = []
        unlabelled_idx = []
        labels         = []

        for i, meta in enumerate(all_metadatas):
            lbl = (meta.get("label") or "unlabelled").strip()
            if lbl and lbl not in ("unlabelled", ""):
                labelled_idx.append(i)
                labels.append(lbl)
            else:
                unlabelled_idx.append(i)

        if len(labelled_idx) < k_neighbors:
            logger.warning(
                f"Only {len(labelled_idx)} labelled samples "
                f"(need >= {k_neighbors}). Skipping propagation."
            )
            return {}

        if not unlabelled_idx:
            logger.info("All templates are already labelled — nothing to propagate")
            return {}

        logger.info(
            f"Label propagation: {len(labelled_idx)} labelled, "
            f"{len(unlabelled_idx)} unlabelled"
        )

        X_labelled   = embeddings[labelled_idx]
        X_unlabelled = embeddings[unlabelled_idx]

        knn = KNeighborsClassifier(
            n_neighbors=min(k_neighbors, len(labelled_idx)),
            metric="cosine",
            n_jobs=-1,
        )
        knn.fit(X_labelled, labels)

        probs            = knn.predict_proba(X_unlabelled)
        predicted_labels = knn.predict(X_unlabelled)
        confidences      = probs.max(axis=1)

        propagated = {}
        updates    = []
        for ul_idx, pred_lbl, conf in zip(
            unlabelled_idx, predicted_labels, confidences
        ):
            if conf >= confidence_threshold:
                tid = all_ids[ul_idx]
                propagated[tid] = pred_lbl
                updates.append((tid, f"propagated:{pred_lbl}", float(conf)))

        # Update ChromaDB in pages
        for start in range(0, len(updates), _PAGE_SIZE):
            page     = updates[start: start + _PAGE_SIZE]
            page_ids = [u[0] for u in page]
            # Map from id -> (label, confidence) for quick lookup
            page_map  = {u[0]: (u[1], u[2]) for u in page}

            try:
                existing  = self.col_templates.get(ids=page_ids, include=["metadatas"])
                ex_ids    = _safe_list(existing.get("ids"))
                ex_metas  = _safe_list(existing.get("metadatas"))
            except Exception as e:
                logger.warning(f"Label update fetch failed: {e}")
                continue

            new_metas = []
            for eid, meta in zip(ex_ids, ex_metas):
                if eid in page_map:
                    meta["label"]                    = page_map[eid][0]
                    meta["propagation_confidence"]   = page_map[eid][1]
                new_metas.append(meta)

            if ex_ids:
                try:
                    self.col_templates.update(ids=ex_ids, metadatas=new_metas)
                except Exception as e:
                    logger.warning(f"Label update write failed: {e}")

        self._bm25_dirty = True
        logger.info(f"Propagated {len(propagated)} labels (threshold={confidence_threshold})")
        return propagated

    # ── Hybrid search ─────────────────────────────────────────────────────────
    def hybrid_search(
        self,
        query_text:      str,
        query_embedding: np.ndarray,
        top_k:           int   = 10,
        semantic_weight: float = 0.7,
        bm25_weight:     float = 0.3,
        collection:      str   = "templates",
    ) -> List[Dict]:
        """[FIX-A, FIX-C, FIX-D] Semantic + BM25 hybrid retrieval."""
        self._ensure_bm25_fresh()

        col = self.col_templates if collection == "templates" else self.col_anomalies
        n   = col.count()
        if n == 0:
            return []

        # Semantic search — [FIX-A] no "ids" in include
        try:
            sem_results = col.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 2, n),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

        # [FIX-D] _safe_nested_list for [[...]] query result shape
        _ids   = _safe_nested_list(sem_results.get("ids"))
        _dists = _safe_nested_list(sem_results.get("distances"))

        sem_scores: Dict[str, float] = {
            doc_id: 1.0 - dist
            for doc_id, dist in zip(_ids, _dists)
        }

        # BM25 search
        bm25_scores_raw: Dict[str, float] = {}
        if self.bm25 and query_text:
            for doc_id, score in self.bm25.query(query_text, top_k=top_k * 2):
                bm25_scores_raw[doc_id] = score

        if bm25_scores_raw:
            max_bm25  = max(bm25_scores_raw.values()) or 1.0
            bm25_norm = {k: v / max_bm25 for k, v in bm25_scores_raw.items()}
        else:
            bm25_norm = {}

        # Combine
        all_ids_set = set(sem_scores) | set(bm25_norm)
        combined    = {
            did: semantic_weight * sem_scores.get(did, 0.0)
                 + bm25_weight   * bm25_norm.get(did,  0.0)
            for did in all_ids_set
        }
        ranked_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
        if not ranked_ids:
            return []

        try:
            fetch    = col.get(ids=ranked_ids, include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"Hybrid search fetch failed: {e}")
            return []

        # [FIX-D]
        fetch_ids   = _safe_list(fetch.get("ids"))
        fetch_docs  = _safe_list(fetch.get("documents"))
        fetch_metas = _safe_list(fetch.get("metadatas"))

        results = []
        for doc_id, doc, meta in zip(fetch_ids, fetch_docs, fetch_metas):
            results.append({
                "id":           doc_id,
                "document":     doc,
                "metadata":     meta,
                "hybrid_score": round(combined.get(doc_id, 0.0), 4),
            })
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results

    # ── RAG context builder ───────────────────────────────────────────────────
    def get_context_for_rag(
        self,
        anomaly_embeddings: List[np.ndarray],
        anomaly_texts:      List[str],
        anomaly_metadata:   List[Dict],
        top_k:              int = 5,
    ) -> str:
        parts = ["=== DETECTED ANOMALIES ==="]
        for i, (emb, text, meta) in enumerate(
            zip(anomaly_embeddings, anomaly_texts, anomaly_metadata), start=1
        ):
            parts.append(
                f"\n[Anomaly {i}]\n"
                f"  Template   : {meta.get('template_id', 'unknown')}\n"
                f"  Service    : {meta.get('service',     'unknown')}\n"
                f"  Timestamp  : {meta.get('timestamp',   'unknown')}\n"
                f"  AE2 Error  : {meta.get('ae2_error', 0):.4f}\n"
                f"  Score      : {meta.get('anomaly_score', 0):.3f}\n"
                f"  Label      : {meta.get('label', 'unlabelled')}\n"
                f"  Log Sample : {text[:300]}"
            )
        parts.append("\n=== RETRIEVED NORMAL BASELINES (for comparison) ===")
        seen = set()
        for emb, text in zip(anomaly_embeddings, anomaly_texts):
            for r in self.hybrid_search(text, emb, top_k=top_k):
                tid = r["id"]
                if tid in seen:
                    continue
                seen.add(tid)
                m = r["metadata"]
                parts.append(
                    f"\n[Baseline: {tid}]\n"
                    f"  Template    : {r['document'][:200]}\n"
                    f"  Label       : {m.get('label', 'unlabelled')}\n"
                    f"  Avg Entropy : {m.get('avg_entropy', 0):.3f}\n"
                    f"  Avg Burst   : {m.get('avg_burstiness', 0):.1f}\n"
                    f"  Error Rate  : {m.get('error_rate', 0):.3f}\n"
                    f"  Is Golden   : {m.get('is_golden', False)}\n"
                    f"  Hybrid Score: {r['hybrid_score']}"
                )
        return "\n".join(parts)

    # ── Statistics ────────────────────────────────────────────────────────────
    def get_statistics(self) -> Dict:
        """[FIX-B] Uses count() (O(1)) — never fetches all rows."""
        return {
            "templates":          self.col_templates.count(),
            "anomalies":          self.col_anomalies.count(),
            "baselines":          self.col_baselines.count(),
            "labelled_templates": self._count_labelled(),
        }

    def _count_labelled(self) -> int:
        """Count labelled templates without loading all metadata rows."""
        # Strategy 1: where= filter (fast, ChromaDB >= 0.4)
        try:
            result = self.col_templates.get(
                where={"label": {"$ne": "unlabelled"}},
                include=[],
            )
            ids = _safe_list(result.get("ids"))
            return len(ids)
        except Exception:
            pass

        # Strategy 2: paginated scan (fallback)
        count = 0
        offset = 0
        while True:
            try:
                batch = self.col_templates.get(
                    include=["metadatas"],
                    limit=_PAGE_SIZE,
                    offset=offset,
                )
            except Exception:
                break
            ids   = _safe_list(batch.get("ids"))
            metas = _safe_list(batch.get("metadatas"))
            if not ids:
                break
            count += sum(
                1 for m in metas
                if (m.get("label") or "unlabelled") != "unlabelled"
            )
            if len(ids) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return count

    # ── Label management ──────────────────────────────────────────────────────
    def label_template(self, template_id: str, label: str):
        """Manually assign a ground-truth label to a template."""
        try:
            existing  = self.col_templates.get(ids=[template_id], include=["metadatas"])
            ex_metas  = _safe_list(existing.get("metadatas"))
            if ex_metas:
                meta = ex_metas[0]
                meta["label"] = label
                self.col_templates.update(ids=[template_id], metadatas=[meta])
                logger.info(f"Labelled {template_id} -> '{label}'")
                self._bm25_dirty = True
            else:
                logger.warning(f"Template {template_id} not found in KB")
        except Exception as e:
            logger.error(f"label_template failed for {template_id}: {e}")
