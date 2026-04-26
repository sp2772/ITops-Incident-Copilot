# step1_llm_client.py
"""
STEP 1 — LLM Client & Cost Tracker
=====================================
Teaches:
  - ChatOpenAI pointed at Google AI Studio (OpenAI-compat endpoint)
  - How LangChain wraps the raw HTTP call
  - Extracting token usage from response metadata
  - Simple JSON cost log (replace with BigQuery in production)

Every other step imports get_llm() from here.
Run: python step1_llm_client.py
"""

import os, json, time
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Approximate Google AI Studio pricing per 1M tokens
COST_PER_1M = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-flash-lite-latest": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash-lite-preview-09-2025": {"input": 0.10, "output": 0.40},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-flash-latest": {"input": 0.30, "output": 2.50},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.1-flash-image-preview": {"input": 0.25, "output": 1.50},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-pro-latest": {"input": 1.25, "output": 10.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3.1-pro-preview-customtools": {"input": 2.00, "output": 12.00},
}

DEFAULT_MODEL  = "gemini-2.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash-lite"
COST_LOG_FILE  = Path("cost_log.json")


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.2) -> ChatOpenAI:
    """
    Factory: returns a ChatOpenAI instance pointed at Google AI Studio.

    We use ChatOpenAI (not ChatGoogleGenerativeAI) because Google AI Studio
    exposes an OpenAI-compatible REST endpoint — same SDK, different backend.

    temperature=0.2  → consistent outputs for classification / triage tasks
    temperature=0.7  → more creative, good for runbook writing
    """
    return ChatOpenAI(
        model=model,
        api_key=GOOGLE_API_KEY,
        base_url=GEMINI_BASE_URL,
        temperature=temperature,
    )


class CostTracker:
    """
    Logs every LLM call to a JSON file with token counts and estimated cost.
    In production this would write to BigQuery / Prometheus / Datadog.
    """

    def __init__(self, log_file: Path = COST_LOG_FILE):
        self.log_file = log_file
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        if self.log_file.exists():
            with open(self.log_file) as f:
                self._entries = json.load(f)

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self._entries, f, indent=2)

    def calculate_cost(self, model: str, prompt_tok: int, completion_tok: int) -> float:
        rates = COST_PER_1M.get(model, COST_PER_1M[DEFAULT_MODEL])
        return round(
            (prompt_tok / 1_000_000) * rates["input"] +
            (completion_tok / 1_000_000) * rates["output"], 8
        )

    def log(self, session_id: str, agent: str, model: str,
            prompt_tok: int, completion_tok: int) -> dict:
        """Record one LLM call and persist to disk."""
        cost = self.calculate_cost(model, prompt_tok, completion_tok)
        entry = {
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id":       session_id,
            "agent":            agent,
            "model":            model,
            "prompt_tokens":    prompt_tok,
            "completion_tokens": completion_tok,
            "cost_usd":         cost,
        }
        self._entries.append(entry)
        self._save()
        print(f"  [Cost] {agent}: {prompt_tok + completion_tok} tok | ${cost:.8f}")
        return entry

    def session_summary(self, session_id: str) -> dict:
        rows = [e for e in self._entries if e["session_id"] == session_id]
        return {
            "session_id":    session_id,
            "calls":         len(rows),
            "total_tokens":  sum(e["prompt_tokens"] + e["completion_tokens"] for e in rows),
            "total_cost_usd": round(sum(e["cost_usd"] for e in rows), 8),
        }


def direct_llm_call(prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Raw LLM call — demonstrates the lowest level before LangChain abstractions.
    response.usage_metadata is LangChain's standardized token field (>=0.3).
    """
    llm  = get_llm(model=model)
    resp = llm.invoke([HumanMessage(content=prompt)])
    usage = resp.usage_metadata or {}
    return {
        "text":              resp.content,
        "prompt_tokens":     usage.get("input_tokens",  0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens":      usage.get("total_tokens",  0),
    }


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uuid

    print("=" * 55)
    print("STEP 1 TEST — LLM Client & Cost Tracker")
    print("=" * 55)

    sid     = f"step1-{uuid.uuid4().hex[:8]}"
    tracker = CostTracker()

    print("\n[Test 1] Basic LLM call")
    r = direct_llm_call("What is Kubernetes? One sentence.")
    print(f"  Response : {r['text']}")
    print(f"  Tokens   : {r['total_tokens']}")

    print("\n[Test 2] Cost logging")
    tracker.log(sid, "test_agent", DEFAULT_MODEL, r["prompt_tokens"], r["completion_tokens"])
    print(f"  Session summary: {tracker.session_summary(sid)}")

    print("\n✅ STEP 1 COMPLETE — cost_log.json written")