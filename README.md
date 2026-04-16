# 🧠 Self-Evolving AIOps Log Intelligence System

> 100% Open Source · Zero Cost · Local Execution · Streams 100M+ Logs

---

# 🚀 AIOps Anomaly Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Anomaly%20Detection-orange)
![RAG](https://img.shields.io/badge/RAG-LLM%20Powered-green)
![Database](https://img.shields.io/badge/DB-DuckDB%20%7C%20ChromaDB-purple)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Overview

An end-to-end **AIOps system** for:
- Log ingestion (any format)
- Feature engineering & streaming
- Multi-stage anomaly detection (ensemble)
- Knowledge base with semantic + keyword retrieval
- Automated incident report generation using LLMs (RAG)

---

## 📊 Architecture

📄 Detailed Diagram: [Architecture](docs/architecture.md)

---

## 🧠 Business Logic

📄 Full Logic: [Business Logic](docs/business_logic.md)

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/vinay170497/AIOPS-Anomaly-Detection-System.git
cd AIOPS-Anomaly-Detection-System

**###2. Setup Environment**
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

**### 3. Install Dependencies**
pip install -r requirements.txt

**### 4. Run Pipeline**
python main.py

**### 5. Outputs**
Anomalies → /outputs/anomalies/
Reports → /outputs/reports/
