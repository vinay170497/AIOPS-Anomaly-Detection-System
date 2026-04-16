# 🧠 Business Logic — AIOps Anomaly Detection System

![ML](https://img.shields.io/badge/Machine%20Learning-Anomaly%20Detection-orange)
![RAG](https://img.shields.io/badge/RAG-LLM%20Powered-green)
![Pipeline](https://img.shields.io/badge/Pipeline-Streaming%20%26%20Batch-blue)
![Status](https://img.shields.io/badge/System-Production%20Ready-success)

---

## 📑 Table of Contents

1. [Objective](#-objective)
2. [System Phases Overview](#-system-phases-overview)
   - [Phase 1 — Streaming Ingestion](#-phase-1--streaming-ingestion)
   - [Phase 2 — Anomaly Sandwich Ensemble](#-phase-2--anomaly-sandwich-ensemble)
   - [Phase 3 — Knowledge Base](#-phase-3--knowledge-base)
   - [Phase 4 — RAG Analyst](#-phase-4--rag-analyst)
3. [End-to-End Flow](#-end-to-end-flow)
4. [Key Design Principles](#-key-design-principles)
5. [Future Enhancements](#-future-enhancements)

---

## 🎯 Objective

Build a **scalable, intelligent AIOps system** capable of:

- Processing logs from **any format and source**
- Detecting anomalies using **multi-stage ML ensemble**
- Persisting knowledge for **continuous learning**
- Generating **human-readable incident reports** using LLMs

---

## 🧩 System Phases Overview

---

### 🔹 Phase 1 — Streaming Ingestion

**Goal:** Transform unstructured logs into structured feature vectors

**Pipeline:**
- Raw Logs → Drain3 → Structured Templates → DuckDB → Feature Engineering

**Feature Signals:**
- Entropy (randomness)
- Burstiness (event spikes)
- Volatility (pattern instability)
- Time Delta (temporal gaps)
- HTTP Signals (status, methods, failures)

**Output:**
📊 Feature Matrix (N × 393)

---

### 🔹 Phase 2 — Anomaly Sandwich Ensemble

**Goal:** Achieve high-precision anomaly detection with layered filtering

#### 🧠 Stage 1 — Autoencoder (AE1)
- Learns global data representation
- Reduces dimensionality:

393 → 384 → 64 (latent space)


#### 🌲 Stage 2 — Isolation Forest
- Removes obvious anomalies
- Produces clean dataset

#### 🧪 Stage 3 — Autoencoder (AE2 - Golden Baseline)
- Trained only on normal data
- Detects subtle anomalies

#### ⚖️ Decision Logic


IF reconstruction_error > threshold:
LABEL = ANOMALY
ELSE:
LABEL = NORMAL


---

### 🔹 Phase 3 — Knowledge Base

**Goal:** Provide contextual intelligence and memory

**Core Components:**
- 🧠 ChromaDB (vector database)
- 🔗 KNN Label Propagation
- 🔍 BM25 Keyword Retrieval

**Capabilities:**
- Store anomaly embeddings
- Retrieve similar incidents
- Improve predictions over time

---

### 🔹 Phase 4 — RAG Analyst

**Goal:** Convert anomalies into actionable insights

**Pipeline:**
1. Hybrid retrieval:
   - Semantic (vector similarity)
   - Keyword (BM25)
2. Context aggregation
3. LLM reasoning (Ollama)
4. Report generation

**Generated Output:**
- Root cause analysis
- Severity classification
- Suggested remediation steps

---

## 🔁 End-to-End Flow

Raw Logs
↓
Feature Engineering
↓
AE1 → Isolation Forest → AE2
↓
Anomaly Detection
↓
Knowledge Base
↓
RAG + LLM
↓
Incident Report


---

## ⚙️ Key Design Principles

- 🧩 **Layered Detection**
  Reduces false positives significantly

- 🔍 **Separation of Concerns**
  Representation ≠ anomaly scoring

- 🔗 **Hybrid Retrieval**
  Prevents hallucination in LLM outputs

- ⚡ **Streaming First Design**
  Enables scalability with minimal infra

- 🧠 **Continuous Learning**
  Knowledge base improves system over time

---

## 🚀 Future Enhancements

- Real-time streaming with Kafka
- Human-in-the-loop feedback system
- Observability dashboard
- Alerting integrations (Slack, Email, Webhooks)