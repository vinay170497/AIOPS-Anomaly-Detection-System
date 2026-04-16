# 📊 System Architecture

> End-to-end flow of the AIOps Anomaly Detection System

```mermaid
flowchart TD

A[Raw Logs Any Format] --> B[Drain3 Templating]
B --> C[DuckDB Streaming Storage]
C --> D[Feature Engineering Features Extraction]
D --> E[Feature Matrix N x 393]

E --> F[Autoencoder AE1 Global Learning]
F --> G[Isolation Forest Filtering]
G --> H[Normal Data Subset]
H --> I[Autoencoder AE2 Golden Baseline]

I --> J{Reconstruction Error Above Threshold}

J -->|Yes| K[Anomaly Detected]
J -->|No| L[Normal Behavior]

K --> M[ChromaDB Knowledge Base]

M --> N[KNN Label Propagation]
M --> O[BM25 Keyword Retrieval]

N --> P[Hybrid Retrieval Layer]
O --> P

P --> Q[Ollama LLM Analysis]
Q --> R[Incident Report Generation]