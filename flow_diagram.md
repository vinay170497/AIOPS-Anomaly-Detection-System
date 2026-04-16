# System Architecture

```mermaid
flowchart TD

A[Raw Logs (Any Format)]
    --> B[Drain3 Templating]

B --> C[DuckDB Streaming Storage]

C --> D[Feature Engineering<br/>Entropy · Burstiness · Volatility · Time Delta · HTTP]

D --> E[Feature Matrix (N × 393)]

E --> F[Autoencoder AE1]
F --> G[Isolation Forest]
G --> H[Normal Subset]
H --> I[Autoencoder AE2]

I --> J{Reconstruction Error > Threshold?}

J -->|Yes| K[ANOMALY]
J -->|No| L[NORMAL]

K --> M[ChromaDB Knowledge Base]

M --> N[KNN Label Propagation]
M --> O[BM25 Retrieval]

N --> P[Hybrid Retrieval]
O --> P

P --> Q[Ollama LLM]
Q --> R[Incident Report]
