```markdown
# 🧠 Business Logic — Log Intelligence & Anomaly Detection System

## 📑 Table of Contents

1. Objective
2. System Phases Overview
   - Phase 1 — Streaming Ingestion
   - Phase 2 — Anomaly Sandwich Ensemble
   - Phase 3 — Knowledge Base
   - Phase 4 — RAG Analyst
3. End-to-End Flow
4. Key Design Principles
5. Future Enhancements

---

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
Feature Matrix (N × 393)

---

### 🔹 Phase 2 — Anomaly Sandwich Ensemble

**Goal:** Detect anomalies with high precision using layered models

#### Step 1: Autoencoder (AE1)
- Learns global representation
- Reduces dimensionality:
  393 → 384 → 64

#### Step 2: Isolation Forest
- Filters point anomalies
- Keeps only normal samples

#### Step 3: Autoencoder (AE2 - Golden Baseline)
- Trained only on normal data
- Computes reconstruction error

#### Decision Logic
IF reconstruction_error > threshold:
LABEL = ANOMALY
ELSE:
LABEL = NORMAL


---

### 🔹 Phase 3 — Knowledge Base

**Goal:** Store and retrieve contextual anomaly information

**Components:**
- ChromaDB (vector storage)
- KNN Label Propagation
- BM25 Retrieval

**Capabilities:**
- Store anomaly embeddings
- Retrieve similar incidents
- Improve labeling over time

---

### 🔹 Phase 4 — RAG Analyst

**Goal:** Generate explainable incident reports

**Pipeline:**
1. Semantic search (vector similarity)
2. BM25 keyword search
3. Merge retrieved context
4. Pass to LLM (Ollama)
5. Generate:
   - Root cause
   - Severity
   - Suggested actions

**Output:**
Structured Incident Report

---

## 🔁 End-to-End Flow

Raw Logs  
→ Feature Engineering  
→ AE1 → Isolation Forest → AE2  
→ Anomaly Detection  
→ Knowledge Base  
→ RAG + LLM  
→ Incident Report  

---

## ⚙️ Key Design Principles

- Layered anomaly detection improves precision
- Separation of representation and scoring
- Hybrid retrieval avoids hallucination
- Streaming design ensures scalability
- Knowledge base enables continuous learning

---

## 🚀 Future Enhancements

- Kafka-based real-time streaming
- Feedback-driven learning loop
- Visualization dashboard
- Alerting system integrations