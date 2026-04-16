# 🧠 Self-Evolving AIOps Log Intelligence System

> 100% Open Source · Zero Cost · Local Execution · Streams 100M+ Logs

---
```markdown
## 📊 Architecture Diagram

See full diagram: [Architecture](flow_diagram.md)

## Architecture Overview

```
Raw Logs (any format)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Streaming Ingestion                                      │
│  Drain3 templating → DuckDB (disk-backed, streaming chunks)         │
│  Features: entropy · burstiness · volatility · time_delta · HTTP   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ feature matrix (N × 393)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Anomaly Sandwich Ensemble                                │
│                                                                     │
│   AE1 ──── learns global representation (384d → 64d latent)        │
│    │                                                                 │
│    ▼ latent space                                                   │
│   Isolation Forest ──── filters point anomalies                     │
│    │                                                                 │
│    ▼ "normal" subset only                                           │
│   AE2 ──── Golden Baseline (trained on normals only)                │
│    │                                                                 │
│    ▼ reconstruction error > threshold = ANOMALY                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Knowledge Base                                           │
│  ChromaDB (PersistentClient) + KNN Label Propagation + BM25        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — RAG Analyst                                              │
│  Hybrid Retrieval (Semantic + BM25) → Ollama LLM → Incident Report │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
aiops_system/
├── miner_config.json        # Drain3 configuration (masking, depth, thresholds)
├── requirements.txt         # All Python dependencies
├── ingestion_engine.py      # Streaming DuckDB ingest (Drain3 + feature parsing)
├── feature_extractor.py     # Entropy · Burstiness · Volatility · Embeddings
├── anomaly_pipeline.py      # AE1 → IF → AE2 PyTorch ensemble
├── knowledge_manager.py     # ChromaDB CRUD + KNN label propagation + BM25
├── analyst_rag.py           # LangChain-style RAG with Ollama LLM
├── main.py                  # CLI entrypoint (ingest / train / analyse / chat)
└── README.md                # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Python 3.9+ required
pip install -r requirements.txt
```

### 2. Install & Start Ollama

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Start the server
ollama serve &

# Pull your preferred model (pick one)
ollama pull deepseek-r1:8b      # Recommended: better reasoning
ollama pull llama3:8b            # Alternative: faster responses
```

### 3. Ingest Logs

```bash
# Single file
python main.py --mode ingest --input /var/log/application.log

# Entire directory
python main.py --mode ingest --input /var/log/

# Resume interrupted ingestion (auto-detects last line)
python main.py --mode ingest --input /var/log/application.log
```

### 4. Train the Pipeline

```bash
# Full training on all ingested logs
python main.py --mode train

# Cap rows for quick testing
python main.py --mode train --max-rows 100000
```

### 5. Analyse New Logs

```bash
# Ingest + detect + RAG report (streaming output)
python main.py --mode analyse --input /var/log/new_errors.log \
    --query "Why is the checkout service throwing 5xx errors?"

# Non-streaming (print full report at end)
python main.py --mode analyse --input /var/log/new_errors.log --no-stream
```

### 6. Interactive Chat

```bash
python main.py --mode chat
# Commands: 'stats' | 'search <query>' | 'exit'
```

### 7. System Status

```bash
python main.py --mode status
```

---

## Configuration Override

Create `config.json` to override any defaults:

```json
{
  "db_path":           "aiops_logs.duckdb",
  "model_dir":         "models",
  "chroma_dir":        "chromadb_store",
  "report_dir":        "reports",
  "chunk_size":        10000,
  "ae1_epochs":        30,
  "ae2_epochs":        40,
  "if_contamination":  0.05,
  "anomaly_percentile": 95.0,
  "ollama_model":      "deepseek-r1:8b",
  "ollama_url":        "http://localhost:11434",
  "knn_k":             5,
  "label_confidence":  0.6
}
```

---

## Manual Label Assignment (Ground Truth)

```python
from knowledge_manager import KnowledgeManager

km = KnowledgeManager(persist_dir="chromadb_store")

# Assign a human-verified label to a template
km.label_template("E42", "DB Connection Timeout")
km.label_template("E17", "Memory Pressure")
km.label_template("E91", "Rate Limiter Triggered")

# Re-run label propagation after adding ground truth
km.propagate_labels(k_neighbors=5, confidence_threshold=0.6)
```

---

## Memory & Performance Notes

| Operation | RAM Usage | Strategy |
|---|---|---|
| Log ingestion | ~200 MB | Streaming chunks via `yield` |
| Feature extraction | ~500 MB per chunk | `batch_embed_size=256` |
| AE training | ~1–2 GB | Mini-batch SGD, early stopping |
| Scoring | ~300 MB per chunk | Chunked inference |
| ChromaDB | Disk-backed | PersistentClient |

> **100M logs** ≈ ~15–20 GB DuckDB file. Feature matrix fits in RAM only for ~5M rows simultaneously — the pipeline handles this via streaming.

---

## Supported Log Formats

- Apache Combined Log Format
- Syslog (RFC 3164/5424)
- Log4j / Logback (Java)
- Generic ISO8601 timestamp + level + message
- Custom formats (add regex patterns to `LOG_PATTERNS` in `ingestion_engine.py`)

---

## Feature Vector Layout (393 dimensions)

```
[  0 : 383]  → Semantic embedding (all-MiniLM-L6-v2, L2-normalised)
[ 384      ]  → Shannon entropy of log message
[ 385      ]  → Burstiness (template freq in 10s sliding window)
[ 386      ]  → Volatility (std-dev of numeric params in template group)
[ 387      ]  → Time delta (seconds since last event in same service)
[ 388      ]  → Log level (ordinal: 0=TRACE → 5=FATAL)
[ 389      ]  → HTTP 1xx flag
[ 390      ]  → HTTP 2xx flag
[ 391      ]  → HTTP 3xx flag
[ 392      ]  → HTTP 4xx flag (client errors)
[ 393-1    ]  → HTTP 5xx flag (server errors)
```

---

## Troubleshooting

**`ollama: command not found`**
→ Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`

**`ConnectionRefusedError` when calling Ollama**
→ Start server: `ollama serve`

**`No features extracted`**
→ Ensure you ran `ingest` mode before `train` mode

**OOM during training**
→ Reduce `chunk_size` and `train_batch_size` in config; or use `--max-rows`

**ChromaDB dimension mismatch**
→ Delete `chromadb_store/` and retrain if you change embedding models
