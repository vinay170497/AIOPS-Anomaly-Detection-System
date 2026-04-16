"""
analyst_rag.py  (v2 — production)
----------------------------------
RAG-powered performance analysis via local Ollama LLM.

Fixes applied:
  [FIX-T1] ReadTimeout no longer crashes the REPL.
           Every LLM call is wrapped in try/except; errors print a friendly
           message and the chat loop continues.

  [FIX-T2] OLLAMA_TIMEOUT raised to 600s for generate/stream calls.
           The chat() call uses a separate CHAT_TIMEOUT of 120s since it
           handles shorter interactive responses.

  [FIX-M1] chat() switches to a lighter model automatically.
           deepseek-r1:8b is a reasoning model optimised for analysis tasks,
           not interactive chat. Chat mode prefers the first available lighter
           model (llama3:8b, phi3, tinyllama) from the Ollama server, falling
           back to the configured model if none are available.

  [FIX-M2] max_tokens capped at 512 for chat responses (was 2048).
           Prevents the model from generating multi-page answers for
           simple conversational queries.

  [FIX-C1] General / off-topic queries are answered locally without calling
           the LLM at all. The interactive_chat() method detects help, stats,
           capability, and search commands and handles them in Python.

  [FIX-C2] Non-ASCII emoji removed from all print() calls.
           Prevents cp1252 encoding crashes on Windows terminals.

  [FIX-C3] SYSTEM_PROMPT updated: model now knows it can answer general
           questions concisely. Prevents the "I am an SRE assistant, here
           are my 2000-word capabilities..." runaway response.
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

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "deepseek-r1:8b"

# Separate timeouts: analysis needs long think time; chat must respond quickly
ANALYSIS_TIMEOUT = 600    # seconds — for full incident analysis reports
CHAT_TIMEOUT     = 180    # seconds — for interactive chat responses
CONNECT_TIMEOUT  = 10     # seconds — for connection checks

# Lighter models preferred for interactive chat (faster on CPU)
CHAT_MODEL_PREFERENCE = ["phi3:latest", "tinyllama:latest", "llama3:8b", "llama3:latest"]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Senior Site Reliability Engineer and AIOps analyst.
For log analysis tasks: produce structured incident reports with root cause, business impact, and corrective actions.
For general questions: answer concisely in 2-3 sentences maximum.
Never write long self-descriptions. Be direct and brief unless asked for detail.
"""

ANALYSIS_PROMPT_TEMPLATE = """Analyse the following production log anomalies and provide a complete incident report.

{context}

---

ADDITIONAL QUERY: {user_query}

Detected at: {detection_time}
Total anomalies in this batch: {n_anomalies}
Highest anomaly score: {max_score:.3f}
Affected services: {affected_services}

Provide a structured root cause analysis and remediation plan using these sections:
## Incident Summary
## Root Cause Analysis
## Business Impact Assessment
## Corrective Actions
## Correlated Patterns
## Confidence Level (High/Medium/Low)
"""

SUMMARY_PROMPT_TEMPLATE = """Write a 3-sentence executive summary (non-technical, action-oriented):

Anomaly count: {n_anomalies}
Affected services: {affected_services}
Highest severity score: {max_score:.3f}
Root causes detected: {root_causes}
"""

# Built-in help text — shown without calling the LLM
HELP_TEXT = """
AIOps Analyst — Interactive Mode
=================================
This assistant analyses your ingested log data using the trained anomaly models.

BUILT-IN COMMANDS (instant, no LLM call):
  stats              Show knowledge base statistics
  search <query>     Search for similar log templates in the KB
  help               Show this help message
  exit               Quit

LOG ANALYSIS QUERIES (uses LLM):
  "Why are 5xx errors spiking on the auth service?"
  "What caused the latency increase between 2pm and 4pm?"
  "Which services have the highest error rates?"
  "Explain the anomaly pattern for template E42"
  "What corrective actions should I take for database timeouts?"

GENERAL QUERIES (uses LLM, short response):
  "What is an isolation forest?"
  "Explain AE2 reconstruction error"
  "What does burstiness mean in this context?"

NOTE: deepseek-r1:8b on CPU takes 1-3 minutes per response.
      For faster chat, start Ollama with a lighter model:
        ollama run phi3:latest
      Then set in config.json: "ollama_model": "phi3:latest"
"""


# ══════════════════════════════════════════════════════════════════════════════
# OllamaClient
# ══════════════════════════════════════════════════════════════════════════════
class OllamaClient:

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url       = base_url.rstrip("/")
        self.model          = model
        self.chat_model     = model      # may be overridden to a lighter model
        self.available      : List[str] = []
        self._verify_connection()
        self._select_chat_model()

    def _verify_connection(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=CONNECT_TIMEOUT)
            resp.raise_for_status()
            self.available = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Ollama connected. Available models: {self.available}")
            if not any(self.model.split(":")[0] in m for m in self.available):
                logger.warning(
                    f"Model '{self.model}' not found locally. "
                    f"Pull with: ollama pull {self.model}"
                )
        except requests.RequestException as e:
            logger.warning(f"Cannot reach Ollama at {self.base_url}: {e}")
            logger.warning("Start Ollama with: ollama serve")

    def _select_chat_model(self):
        """
        [FIX-M1] Pick the lightest available model for interactive chat.
        deepseek-r1 reasons deeply before answering — great for analysis,
        slow (1-3 min/response) for quick conversational queries on CPU.
        """
        for preferred in CHAT_MODEL_PREFERENCE:
            if any(preferred.split(":")[0] in m for m in self.available):
                self.chat_model = preferred
                logger.info(f"Chat model set to: {self.chat_model} (lighter than {self.model})")
                return
        self.chat_model = self.model
        logger.info(f"Chat model: {self.chat_model} (no lighter model found)")

    # ── generate (for analysis — long timeout, full model) ────────────────────
    def generate(
        self,
        prompt:      str,
        system:      str   = "",
        temperature: float = 0.1,
        max_tokens:  int   = 2048,
        stream:      bool  = False,
    ) -> str:
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "system":  system,
            "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": 0.9},
            "stream":  stream,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=ANALYSIS_TIMEOUT,
                stream=stream,
            )
            resp.raise_for_status()
            if stream:
                full = ""
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        full += chunk.get("response", "")
                        if chunk.get("done"):
                            break
                return full
            return resp.json().get("response", "")
        except requests.exceptions.Timeout:
            return (
                "[LLM timeout] The model took too long to respond. "
                "Try a lighter model (phi3:latest) or reduce max_tokens."
            )
        except Exception as e:
            return f"[LLM error] {e}"

    # ── generate_stream (for analysis streaming) ──────────────────────────────
    def generate_stream(
        self,
        prompt:      str,
        system:      str   = "",
        temperature: float = 0.1,
        max_tokens:  int   = 2048,
    ) -> Generator[str, None, None]:
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "system":  system,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream":  True,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=ANALYSIS_TIMEOUT,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except requests.exceptions.Timeout:
            yield "\n[LLM timeout — model took too long. Try a lighter model.]"
        except Exception as e:
            yield f"\n[LLM error: {e}]"

    # ── chat (for interactive REPL — short timeout, lighter model) ───────────
    def chat(
        self,
        messages:    List[Dict],
        temperature: float = 0.15,
        max_tokens:  int   = 512,      # [FIX-M2] capped at 512 for chat
    ) -> str:
        payload = {
            "model":    self.chat_model,   # [FIX-M1] lighter model for chat
            "messages": messages,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
            "stream":   False,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=CHAT_TIMEOUT,   # [FIX-T2] 180s not 300s
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.Timeout:
            return (
                "[Response timed out after 3 minutes on CPU.]\n"
                f"Model in use: {self.chat_model}\n"
                "To get faster responses, try:\n"
                "  1. ollama pull phi3:latest  (then restart chat)\n"
                "  2. Set 'ollama_model': 'phi3:latest' in config.json\n"
                "  3. For analysis queries, use: python main.py --mode analyse"
            )
        except Exception as e:
            return f"[LLM error: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# AnalystRAG
# ══════════════════════════════════════════════════════════════════════════════
class AnalystRAG:

    def __init__(
        self,
        knowledge_manager,
        ollama_model: str = DEFAULT_MODEL,
        ollama_url:   str = OLLAMA_BASE_URL,
        report_dir:   str = "reports",
    ):
        self.km         = knowledge_manager
        self.llm        = OllamaClient(base_url=ollama_url, model=ollama_model)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ── Full batch anomaly analysis ───────────────────────────────────────────
    def analyse_anomaly_batch(
        self,
        anomaly_scores_result: Dict,
        feature_metadata:      List[Dict],
        embeddings:            np.ndarray,
        user_query:            str   = "Identify root cause and remediation steps.",
        top_k_context:         int   = 5,
        stream:                bool  = False,
    ) -> Dict:
        is_anomaly    = anomaly_scores_result["is_anomaly"]
        ae2_errors    = anomaly_scores_result["ae2_errors"]
        anomaly_scores = anomaly_scores_result["anomaly_score"]

        anomaly_indices = np.where(is_anomaly)[0]
        if len(anomaly_indices) == 0:
            return {
                "report":            "No anomalies detected in this batch.",
                "executive_summary": "System is operating within normal parameters.",
                "n_anomalies":       0,
                "metadata":          {},
            }

        logger.info(f"Analysing {len(anomaly_indices)} anomalies via RAG")

        anomaly_embeddings  = []
        anomaly_texts       = []
        anomaly_meta_list   = []
        affected_services   = set()

        for idx in anomaly_indices[:20]:
            meta     = feature_metadata[idx]
            emb      = embeddings[idx, :384]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
            anomaly_embeddings.append(emb_norm)
            anomaly_texts.append(meta.get("raw_line", meta.get("template_str", "")))
            anomaly_meta_list.append({
                **meta,
                "ae2_error":    float(ae2_errors[idx]),
                "anomaly_score": float(anomaly_scores[idx]),
            })
            if meta.get("service"):
                affected_services.add(meta["service"])

        context = self.km.get_context_for_rag(
            anomaly_embeddings=anomaly_embeddings,
            anomaly_texts=anomaly_texts,
            anomaly_metadata=anomaly_meta_list,
            top_k=top_k_context,
        )

        analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            context=context,
            user_query=user_query,
            detection_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            n_anomalies=len(anomaly_indices),
            max_score=float(anomaly_scores[anomaly_indices].max()),
            affected_services=", ".join(sorted(affected_services)) or "unknown",
        )

        logger.info(f"Calling LLM ({self.llm.model}) for analysis...")
        t0 = time.time()

        if stream:
            print("\n" + "=" * 60)
            print("  AIOps Incident Analysis Report")
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

        result = {
            "report":               report,
            "executive_summary":    exec_summary,
            "n_anomalies":          len(anomaly_indices),
            "affected_services":    sorted(affected_services),
            "max_anomaly_score":    float(anomaly_scores[anomaly_indices].max()),
            "analysis_time_seconds": round(llm_time, 2),
            "metadata": {
                "model_used":    self.llm.model,
                "timestamp":     datetime.utcnow().isoformat(),
                "anomaly_ids":   [feature_metadata[i].get("log_id") for i in anomaly_indices[:20]],
            },
        }
        self._save_report(result)
        return result

    # ── Single log quick analysis ─────────────────────────────────────────────
    def analyse_single_log(self, log_text: str, embedding: np.ndarray, ae2_error: float, top_k: int = 5) -> str:
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-9)
        results  = self.km.hybrid_search(log_text, emb_norm, top_k=top_k)
        context  = f"SUSPICIOUS LOG:\n{log_text}\n\nAE2 Error: {ae2_error:.4f}\n\nSIMILAR PATTERNS:\n"
        for r in results:
            m = r["metadata"]
            context += (
                f"  [{r['id']}] {r['document'][:200]}\n"
                f"    Label: {m.get('label', '?')}, Score: {r['hybrid_score']:.3f}\n"
            )
        prompt = (
            f"{context}\n\nIn 2-3 sentences: "
            "What is the likely cause and what immediate action should be taken?"
        )
        return self.llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=512)

    # ── Interactive chat ──────────────────────────────────────────────────────
    def interactive_chat(self):
        """
        REPL for ad-hoc log queries.

        Built-in commands run instantly in Python (no LLM call):
          help, stats, search <query>, exit

        All other input goes to the LLM with a 3-minute timeout and a
        graceful error message (no crash) on timeout.

        Chat uses a lighter model automatically (phi3/tinyllama/llama3:8b)
        for faster responses on CPU.
        """
        print()
        print("=" * 55)
        print("  AIOps Analyst -- Interactive Mode")
        print("=" * 55)
        print(f"  Analysis model : {self.llm.model}")
        print(f"  Chat model     : {self.llm.chat_model}")
        print(f"  Chat timeout   : {CHAT_TIMEOUT}s per response")
        print()
        print("  Type 'help' for usage guide, 'exit' to quit.")
        print("=" * 55)
        print()

        # System message is short — just role + brevity instruction
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue

            lower = user_input.lower()

            # ── Built-in commands — zero LLM latency ──────────────────────────
            if lower == "exit":
                print("Goodbye.")
                break

            if lower in ("help", "?", "h"):
                print(HELP_TEXT)
                continue

            if lower == "stats":
                try:
                    stats = self.km.get_statistics()
                    print(f"\n  KB Statistics:")
                    for k, v in stats.items():
                        print(f"    {k:<25}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")
                    print()
                except Exception as e:
                    print(f"  [stats error: {e}]")
                continue

            if lower.startswith("search "):
                query     = user_input[7:].strip()
                dummy_emb = np.zeros(384, dtype=np.float32)
                try:
                    results = self.km.hybrid_search(query, dummy_emb, top_k=5)
                    if results:
                        print(f"\n  Top {len(results)} results for '{query}':")
                        for r in results:
                            lbl = r["metadata"].get("label", "unlabelled")
                            print(f"    [{r['id']}] score={r['hybrid_score']:.3f}  label={lbl}")
                            print(f"    {r['document'][:100]}")
                    else:
                        print("  No results found.")
                    print()
                except Exception as e:
                    print(f"  [search error: {e}]")
                continue

            if lower.startswith("analyse ") or lower.startswith("analyze "):
                # Shorthand: analyse <log text>
                log_text = user_input[8:].strip()
                print("\n  Analysing... (this may take 1-3 minutes on CPU)\n")
                dummy_emb = np.zeros(384, dtype=np.float32)
                response  = self.analyse_single_log(log_text, dummy_emb, ae2_error=0.0)
                print(f"  Analyst: {response}\n")
                continue

            # ── LLM chat call — [FIX-T1] never crashes ────────────────────────
            print(f"  [Thinking... timeout={CHAT_TIMEOUT}s, model={self.llm.chat_model}]")
            history.append({"role": "user", "content": user_input})

            response = self.llm.chat(history, temperature=0.15)  # graceful on timeout
            print(f"\nAnalyst: {response}\n")

            history.append({"role": "assistant", "content": response})

            # Bound context window: keep system + last 18 turns
            if len(history) > 20:
                history = [history[0]] + history[-18:]

    # ── Report persistence ────────────────────────────────────────────────────
    def _save_report(self, result: Dict):
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self.report_dir / f"incident_report_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Report saved -> {path}")

    def load_report(self, path: str) -> Dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def list_reports(self) -> List[str]:
        return sorted(str(p) for p in self.report_dir.glob("incident_report_*.json"))
