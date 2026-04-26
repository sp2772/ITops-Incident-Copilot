# main.py
"""
MAIN — Full IT Ops Copilot Orchestration
==========================================
Imports and wires all 5 steps together:
  step1 → LLM client + cost tracking
  step2 → LangChain tools
  step3 → ReAct agents
  step4 → LangGraph workflow
  step5 → FastAPI app

Run: python main.py
  → Prints a full demo workflow result
  → Then starts FastAPI on http://localhost:8080
  → Django (step below) calls this API

This is the "production entry point" — in a real deployment you'd
run this with: uvicorn main:app --workers 4
"""

import uuid, time, threading
import uvicorn

# Import our modules (each was tested independently in steps 1-5)
from step1_llm_client import CostTracker, DEFAULT_MODEL
from step4_langgraph_workflow import IncidentWorkflow
from step5_fastapi import app  # the FastAPI app object


def run_demo():
    """
    Run a full workflow demo before starting the server.
    Prints results to console so you can verify everything works
    end-to-end before accepting real traffic.
    """
    print("\n" + "=" * 60)
    print("IT OPS COPILOT — Full System Demo")
    print("=" * 60)

    tracker  = CostTracker()
    workflow = IncidentWorkflow()
    session  = f"demo-{uuid.uuid4().hex[:6]}"

    incidents = [
        {
            "id":   f"INC-{uuid.uuid4().hex[:6].upper()}",
            "text": "CRITICAL: Production Kubernetes cluster — all pods in CrashLoopBackOff. "
                    "OOMKilled errors in logs. 100% of users affected. Revenue-impacting.",
        },
        {
            "id":   f"INC-{uuid.uuid4().hex[:6].upper()}",
            "text": "API gateway latency increased from 200ms to 2s. "
                    "Intermittent 504 timeouts observed. About 15% of requests failing.",
        },
    ]

    for inc in incidents:
        print(f"\n{'─'*50}")
        print(f"Processing: {inc['id']}")
        result = workflow.run(incident_id=inc["id"], incident_text=inc["text"])

        print(f"  Severity    : {result['severity']}")
        print(f"  Category    : {result['category']}")
        print(f"  Escalated   : {result['is_escalated']}")
        print(f"  QA Verdict  :\n{result['qa_result']}")
        print(f"  Runbook (preview):\n{result['resolution_runbook'][:300]}...")

    print(f"\n{'─'*50}")
    print("Demo complete ✅")
    print(f"Cost log: {tracker.log_file.resolve()}")


def start_server():
    """Start FastAPI in the main thread after demo."""
    print("\n" + "=" * 60)
    print("Starting FastAPI server on http://0.0.0.0:8080")
    print("Interactive docs: http://localhost:8080/docs")
    print("Django frontend should call this API")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    # 1. Run the demo to verify everything works
    #run_demo()

    # 2. Start the production API server
    start_server()