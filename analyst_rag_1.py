"""
analyst_rag.py
---------------
RAG-powered performance analysis using:
  • Local LLM via Ollama (DeepSeek-R1-8B or Llama-3-8B)
  • ChromaDB hybrid retrieval context from KnowledgeManager
  • Structured prompt engineering for root cause analysis
  • Streaming + non-streaming output modes
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np
import requests

logger = logging.getLogger("AnalystRAG")

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
DEFAULT_MODEL    = "deepseek-r1:8b"     # fallback: llama3:8b
OLLAMA_TIMEOUT   = 300                   # seconds

# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Senior Site Reliability Engineer and Performance Analysis Expert.
Your task is to analyse server log anomalies and produce actionable incident reports.

You have access to:
1. A list of DETECTED ANOMALIES with their anomaly scores, templates, and services
2. RETRIEVED NORMAL BASELINES showing what healthy behaviour looks like for similar patterns
3. Historical metric comparisons (entropy, burstiness, error rates)

When analysing, you MUST:
- Identify the most likely ROOT CAUSE based on the anomaly patterns
- Assess BUSINESS IMPACT (severity: P1/P2/P3, affected services, potential data loss)
- Suggest CORRECTIVE ACTIONS (immediate mitigations + long-term fixes)
- Flag any CORRELATED ANOMALIES that suggest a cascading failure
- Be concise and actionable — output a structured incident report

Output format:
## 🔴 Incident Summary
## 🔍 Root Cause Analysis
## 💥 Business Impact Assessment
## 🛠 Corrective Actions
## 🔗 Correlated Patterns
## 📊 Confidence Level (High/Medium/Low)
"""

ANALYSIS_PROMPT_TEMPLATE = """Analyse the following production log anomalies and provide a complete incident report.

{context}

---

ADDITIONAL QUERY: {user_query}

Detected at: {detection_time}
Total anomalies in this batch: {n_anomalies}
Highest anomaly score: {max_score:.3f}
Affected services: {affected_services}

Please provide a structured root cause analysis and remediation plan.
"""

SUMMARY_PROMPT_TEMPLATE = """Given the following anomaly detection results, write a 3-sentence executive summary 
for a non-technical stakeholder:

Anomaly count: {n_anomalies}
Affected services: {affected_services}  
Highest severity score: {max_score:.3f}
Root causes detected: {root_causes}

Keep it brief, jargon-free, and action-oriented.
"""


# ── Ollama client ─────────────────────────────────────────────────────────────
class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._verify_connection()

    def _verify_connection(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Ollama connected. Available models: {available}")
            if not any(self.model.split(":")[0] in m for m in available):
                logger.warning(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
        except requests.RequestException as e:
            logger.warning(
                f"Could not connect to Ollama at {self.base_url}: {e}\n"
                "Ensure Ollama is running: `ollama serve`"
            )

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str:
        """Non-streaming generation."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
            },
            "stream": stream,
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
            stream=stream,
        )
        resp.raise_for_status()

        if stream:
            full_response = ""
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    full_response += token
                    if chunk.get("done", False):
                        break
            return full_response
        else:
            return resp.json().get("response", "")

    def generate_stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Streaming token generator — yields tokens as they arrive."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": True,
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Chat completion endpoint (messages format)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ── RAG Analyst ───────────────────────────────────────────────────────────────
class AnalystRAG:
    """
    Orchestrates the full RAG pipeline:
      anomaly batch → context retrieval → LLM analysis → structured report
    """

    def __init__(
        self,
        knowledge_manager,
        ollama_model: str = DEFAULT_MODEL,
        ollama_url: str = OLLAMA_BASE_URL,
        report_dir: str = "reports",
    ):
        self.km = knowledge_manager
        self.llm = OllamaClient(base_url=ollama_url, model=ollama_model)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ── Core analysis method ──────────────────────────────────────────────────
    def analyse_anomaly_batch(
        self,
        anomaly_scores_result: Dict,
        feature_metadata: List[Dict],
        embeddings: np.ndarray,
        user_query: str = "Identify root cause and remediation steps.",
        top_k_context: int = 5,
        stream: bool = False,
    ) -> Dict:
        """
        Full RAG analysis on a batch of detected anomalies.

        Args:
            anomaly_scores_result: Output from AnomalyPipeline.score()
            feature_metadata:      Metadata list from FeatureExtractor.process_batch()
            embeddings:            Feature matrix (N, 393)
            user_query:            Analyst's specific question
            top_k_context:        How many baseline docs to retrieve per anomaly
            stream:               Stream LLM tokens to stdout

        Returns:
            Dict with 'report', 'executive_summary', 'metadata'
        """
        is_anomaly = anomaly_scores_result["is_anomaly"]
        ae2_errors = anomaly_scores_result["ae2_errors"]
        anomaly_scores = anomaly_scores_result["anomaly_score"]

        # Filter to anomalous rows only
        anomaly_indices = np.where(is_anomaly)[0]
        if len(anomaly_indices) == 0:
            return {
                "report": "✅ No anomalies detected in this batch.",
                "executive_summary": "System is operating within normal parameters.",
                "n_anomalies": 0,
                "metadata": {},
            }

        logger.info(f"Analysing {len(anomaly_indices)} anomalies via RAG")

        # Prepare anomaly data for context
        anomaly_embeddings = []
        anomaly_texts = []
        anomaly_meta_list = []
        affected_services = set()

        for idx in anomaly_indices[:20]:  # cap at 20 anomalies per RAG call
            meta = feature_metadata[idx]
            emb = embeddings[idx, :384]  # semantic part only
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)

            anomaly_embeddings.append(emb_norm)
            anomaly_texts.append(meta.get("raw_line", meta.get("template_str", "")))
            anomaly_meta_list.append({
                **meta,
                "ae2_error": float(ae2_errors[idx]),
                "anomaly_score": float(anomaly_scores[idx]),
            })
            if meta.get("service"):
                affected_services.add(meta["service"])

        # ── Retrieve context from KB ──
        context = self.km.get_context_for_rag(
            anomaly_embeddings=anomaly_embeddings,
            anomaly_texts=anomaly_texts,
            anomaly_metadata=anomaly_meta_list,
            top_k=top_k_context,
        )

        # ── Build prompt ──
        analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            context=context,
            user_query=user_query,
            detection_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            n_anomalies=len(anomaly_indices),
            max_score=float(anomaly_scores[anomaly_indices].max()),
            affected_services=", ".join(sorted(affected_services)) or "unknown",
        )

        # ── LLM call ──
        logger.info(f"Calling LLM ({self.llm.model}) for analysis...")
        t0 = time.time()

        if stream:
            print("\n" + "=" * 60)
            print("📊 AIOps Incident Analysis Report")
            print("=" * 60 + "\n")
            report = ""
            for token in self.llm.generate_stream(
                analysis_prompt, system=SYSTEM_PROMPT, temperature=0.1
            ):
                print(token, end="", flush=True)
                report += token
            print("\n")
        else:
            report = self.llm.generate(
                analysis_prompt, system=SYSTEM_PROMPT, temperature=0.1
            )

        llm_time = time.time() - t0
        logger.info(f"LLM analysis completed in {llm_time:.1f}s")

        # ── Executive summary (separate short call) ──
        labels = list({
            m.get("label", "unknown")
            for m in anomaly_meta_list
            if m.get("label") and m.get("label") != "unlabelled"
        })
        exec_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            n_anomalies=len(anomaly_indices),
            affected_services=", ".join(sorted(affected_services)) or "unknown",
            max_score=float(anomaly_scores[anomaly_indices].max()),
            root_causes=", ".join(labels) if labels else "Under investigation",
        )
        exec_summary = self.llm.generate(exec_prompt, temperature=0.05, max_tokens=256)

        # ── Build result ──
        result = {
            "report": report,
            "executive_summary": exec_summary,
            "n_anomalies": len(anomaly_indices),
            "affected_services": sorted(affected_services),
            "max_anomaly_score": float(anomaly_scores[anomaly_indices].max()),
            "analysis_time_seconds": round(llm_time, 2),
            "metadata": {
                "model_used": self.llm.model,
                "timestamp": datetime.utcnow().isoformat(),
                "anomaly_ids": [
                    feature_metadata[i].get("log_id") for i in anomaly_indices[:20]
                ],
            },
        }

        # ── Auto-save report ──
        self._save_report(result)
        return result

    # ── Single log analysis ───────────────────────────────────────────────────
    def analyse_single_log(
        self,
        log_text: str,
        embedding: np.ndarray,
        ae2_error: float,
        top_k: int = 5,
    ) -> str:
        """Quick analysis for a single suspicious log entry."""
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-9)
        results = self.km.hybrid_search(log_text, embedding_norm, top_k=top_k)

        context = f"SUSPICIOUS LOG:\n{log_text}\n\nAE2 Reconstruction Error: {ae2_error:.4f}\n\n"
        context += "SIMILAR KNOWN PATTERNS:\n"
        for r in results:
            m = r["metadata"]
            context += (
                f"  [{r['id']}] {r['document'][:200]}\n"
                f"    Label: {m.get('label', 'unknown')}, "
                f"Entropy: {m.get('avg_entropy', 0):.3f}, "
                f"Score: {r['hybrid_score']:.3f}\n"
            )

        prompt = (
            f"{context}\n\nBased on the above, in 2-3 sentences: "
            "What is the likely cause of this log anomaly and what immediate action should be taken?"
        )
        return self.llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=512)

    # ── Interactive chat mode ─────────────────────────────────────────────────
    def interactive_chat(self):
        """
        REPL-style interactive chat for ad-hoc log queries.
        Type 'exit' to quit.
        """
        print("\n🤖 AIOps Analyst — Interactive Mode")
        print("Commands: 'stats' | 'search <query>' | 'exit'\n")
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() == "exit":
                break
            if user_input.lower() == "stats":
                print(f"KB Stats: {self.km.get_statistics()}")
                continue
            if user_input.lower().startswith("search "):
                query = user_input[7:]
                dummy_emb = np.zeros(384, dtype=np.float32)
                results = self.km.hybrid_search(query, dummy_emb, top_k=5)
                for r in results:
                    print(f"  [{r['id']}] {r['document'][:120]} (score: {r['hybrid_score']:.3f})")
                continue

            history.append({"role": "user", "content": user_input})
            response = self.llm.chat(history, temperature=0.15)
            print(f"\nAnalyst: {response}\n")
            history.append({"role": "assistant", "content": response})

            # Keep context window bounded
            if len(history) > 20:
                history = [history[0]] + history[-18:]

    # ── Report persistence ────────────────────────────────────────────────────
    def _save_report(self, result: Dict):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self.report_dir / f"incident_report_{ts}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Report saved → {path}")

    def load_report(self, report_path: str) -> Dict:
        with open(report_path) as f:
            return json.load(f)

    def list_reports(self) -> List[str]:
        return sorted(str(p) for p in self.report_dir.glob("incident_report_*.json"))


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick connectivity test — does NOT require a running pipeline.
    Just verifies Ollama connectivity and prompt rendering.
    """
    print("=== AnalystRAG Smoke Test ===")
    try:
        client = OllamaClient()
        response = client.generate(
            "Say 'AIOps analyst online' in exactly 5 words.",
            max_tokens=20,
        )
        print(f"LLM response: {response}")
        print("✓ Ollama connectivity OK")
    except Exception as e:
        print(f"⚠ Ollama not available: {e}")
        print("  Start Ollama with: ollama serve")
        print("  Pull model with:   ollama pull deepseek-r1:8b")
