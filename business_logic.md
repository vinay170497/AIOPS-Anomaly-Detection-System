# 🧠 Business Logic — Log Intelligence & Anomaly Detection System

## 🎯 Objective
Design a scalable system that:
- Ingests raw logs (any format)
- Detects anomalies using a multi-stage ensemble
- Builds a knowledge base of incidents
- Generates human-readable incident reports using RAG + LLM

---

## 🧩 System Phases Overview

### 🔹 Phase 1 — Streaming Ingestion

**Goal:** Convert unstructured logs into structured feature vectors

**Process:**
1. Accept raw logs (JSON, text, syslogs, etc.)
2. Apply Drain3 templating to extract log patterns
3. Stream logs into DuckDB (disk-backed storage)
4. Perform feature engineering:
   - Entropy
   - Burstiness
   - Volatility
   - Time delta
   - HTTP-specific features

**Output:**
- Feature Matrix (N × 393)

---

### 🔹 Phase 2 — Anomaly Sandwich Ensemble

**Goal:** Detect anomalies with high precision using layered models

#### Step 1: Autoencoder (AE1)
- Learns global representation of data
- Reduces dimensionality:
  393 → 384 → 64 (latent space)

#### Step 2: Isolation Forest
- Identifies point anomalies
- Filters dataset to retain only "normal" samples

#### Step 3: Autoencoder (AE2 - Golden Baseline)
- Trained only on clean/normal data
- Computes reconstruction error

#### Decision Logic:
IF reconstruction_error > threshold:
LABEL = ANOMALY
ELSE:
LABEL = NORMAL



---

### 🔹 Phase 3 — Knowledge Base

**Goal:** Store and retrieve contextual anomaly information

**Components:**
- ChromaDB (persistent vector database)
- KNN-based label propagation
- BM25 keyword-based retrieval

**Capabilities:**
- Store embeddings of anomalies
- Retrieve similar past incidents
- Improve labeling over time

---

### 🔹 Phase 4 — RAG Analyst

**Goal:** Generate explainable incident reports

**Pipeline:**
1. Hybrid retrieval:
   - Semantic search (vector similarity)
   - BM25 keyword search
2. Merge retrieved context
3. Pass context to LLM (Ollama)
4. Generate:
   - Root cause
   - Severity
   - Suggested remediation

**Output:**
- Structured Incident Report

---

## 🔁 End-to-End Flow


---

## ⚙️ Key Design Principles

- **Layered Detection (Sandwich Model):**
  Improves precision and reduces false positives

- **Separation of Concerns:**
  Representation learning ≠ anomaly scoring

- **Hybrid Retrieval:**
  Combines semantic + keyword search for accuracy

- **Streaming Architecture:**
  Enables scalability without heavy infrastructure

- **Self-Improving System:**
  Knowledge base enhances future predictions

---

## 🚀 Future Enhancements

- Real-time streaming via Kafka
- Feedback loop for human-in-the-loop learning
- Dashboard for anomaly visualization
- Alerting system (Slack/Email/Webhooks)
