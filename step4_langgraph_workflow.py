# step4_langgraph_workflow.py
"""
STEP 4 — LangGraph Multi-Agent Workflow
=========================================
Teaches:
  - TypedDict state: the shared blackboard all nodes read/write
  - StateGraph: declare nodes (functions) and edges (flow)
  - Conditional edges: dynamic routing based on state values
  - START / END: built-in terminal nodes
  - compile(): validates graph and returns an invokable Runnable

Graph structure:
  START
    │
    ▼
  [triage_node]  ──→  extract severity from text
    │
    ▼  (conditional)
  ┌─────────────────────────────┐
  │ P1 → [escalate_node]        │
  │ P2/P3/P4 → [diagnostic_node]│
  └─────────────────────────────┘
    │
    ▼
  [resolution_node]
    │
    ▼
  [qa_node]
    │
    ▼
   END

Run: python step4_langgraph_workflow.py
"""

import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from step1_llm_client import CostTracker, DEFAULT_MODEL
from step3_lc_agents import TriageAgent, DiagnosticAgent, ResolutionAgent, QAAgent


# ── Shared State ──────────────────────────────────────────────────────────── #
class IncidentState(TypedDict):
    """
    The shared blackboard passed between every node.
    Each node reads what it needs and writes its output back.
    LangGraph passes this dict between nodes — think of it as a pipeline context.
    """
    incident_id:        str
    incident_text:      str
    triage_result:      str
    severity:           str          # Extracted from triage: P1/P2/P3/P4
    category:           str
    diagnostic_result:  str
    resolution_runbook: str
    qa_result:          str
    is_escalated:       bool
    error:              Optional[str]


# ── Node Functions ────────────────────────────────────────────────────────── #
# Each node receives the full state and returns a dict of keys to update.
# LangGraph merges the returned dict into state automatically.

def triage_node(state: IncidentState) -> dict:
    """Node 1: Run TriageAgent, extract severity and category from its output."""
    print(f"  [Graph] triage_node → incident_id={state['incident_id']}")
    agent  = TriageAgent()
    result = agent.run(state["incident_text"])

    # Extract structured fields from agent text output using simple parsing
    severity = "P3"  # default
    category = "general"
    for line in result.splitlines():
        if line.startswith("SEVERITY:") or "severity" in line.lower():
            m = re.search(r"P[1-4]", line)
            if m:
                severity = m.group()
        if line.startswith("CATEGORY:") or "category" in line.lower():
            category = line.split(":", 1)[-1].strip().lower()

    return {
        "triage_result": result,
        "severity":      severity,
        "category":      category,
        "is_escalated":  severity == "P1",
    }


def escalate_node(state: IncidentState) -> dict:
    """
    Node 2a (P1 only): Prepend an escalation notice and page on-call.
    In production: call PagerDuty API / send Slack alert here.
    """
    print(f"  [Graph] escalate_node → P1 INCIDENT — paging on-call")
    notice = (
        "!! ESCALATION TRIGGERED — P1 INCIDENT !!\n"
        "On-call engineer paged via PagerDuty.\n"
        "War room created: #incident-response\n\n"
    )
    # Still need diagnostic — append notice to triage result
    return {"triage_result": notice + state["triage_result"]}


def diagnostic_node(state: IncidentState) -> dict:
    """Node 2b (P2/P3/P4): Run DiagnosticAgent to find root cause."""
    print(f"  [Graph] diagnostic_node → severity={state['severity']}")
    agent  = DiagnosticAgent()
    result = agent.run(state["incident_text"])
    return {"diagnostic_result": result}


def resolution_node(state: IncidentState) -> dict:
    """Node 3: Run ResolutionAgent with full context — triage + diagnosis."""
    print(f"  [Graph] resolution_node")
    agent   = ResolutionAgent()
    context = (
        f"Incident: {state['incident_text']}\n\n"
        f"Triage:\n{state['triage_result']}\n\n"
        f"Diagnosis:\n{state.get('diagnostic_result', 'N/A (P1 fast-path)')}"
    )
    result = agent.run(context)
    return {"resolution_runbook": result}


def qa_node(state: IncidentState) -> dict:
    """Node 4: QAAgent validates the runbook before it is written to the ticket."""
    print(f"  [Graph] qa_node")
    agent  = QAAgent()
    result = agent.run(state["resolution_runbook"])
    return {"qa_result": result}


# ── Conditional Router ────────────────────────────────────────────────────── #
def route_after_triage(state: IncidentState) -> str:
    """
    Called after triage_node. Returns the name of the NEXT node.
    This is how LangGraph does branching — return a node name string.
    """
    if state["severity"] == "P1":
        return "escalate_node"   # → escalate first, then diagnostic
    return "diagnostic_node"    # → standard path


# ── Workflow Class ────────────────────────────────────────────────────────── #
class IncidentWorkflow:
    """
    Encapsulates the compiled LangGraph.
    Build once in __init__, invoke many times with .run().
    """

    def __init__(self):
        self.graph = self._build()

    def _build(self) -> object:
        g = StateGraph(IncidentState)

        # Register nodes
        g.add_node("triage_node",     triage_node)
        g.add_node("escalate_node",   escalate_node)
        g.add_node("diagnostic_node", diagnostic_node)
        g.add_node("resolution_node", resolution_node)
        g.add_node("qa_node",         qa_node)

        # Entry point
        g.add_edge(START, "triage_node")

        # Conditional branch after triage
        g.add_conditional_edges(
            "triage_node",
            route_after_triage,
            {
                "escalate_node":   "escalate_node",
                "diagnostic_node": "diagnostic_node",
            }
        )

        # Escalation path rejoins at diagnostic
        g.add_edge("escalate_node",   "diagnostic_node")

        # Linear tail: diagnostic → resolution → qa → END
        g.add_edge("diagnostic_node", "resolution_node")
        g.add_edge("resolution_node", "qa_node")
        g.add_edge("qa_node",         END)

        return g.compile()

    def run(self, incident_id: str, incident_text: str) -> IncidentState:
        """
        Invoke the compiled graph. Returns the final state dict
        with all agent outputs populated.
        """
        initial_state: IncidentState = {
            "incident_id":        incident_id,
            "incident_text":      incident_text,
            "triage_result":      "",
            "severity":           "",
            "category":           "",
            "diagnostic_result":  "",
            "resolution_runbook": "",
            "qa_result":          "",
            "is_escalated":       False,
            "error":              None,
        }
        print(f"\n[Workflow] Starting incident_id={incident_id}")
        return self.graph.invoke(initial_state)


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uuid

    print("=" * 55)
    print("STEP 4 TEST — LangGraph Multi-Agent Workflow")
    print("=" * 55)

    workflow = IncidentWorkflow()

    # Test A: P3 incident (standard path)
    print("\n── Test A: P3 Incident (standard path) ──")
    result_a = workflow.run(
        incident_id   = f"INC-{uuid.uuid4().hex[:6].upper()}",
        incident_text = "API gateway returning intermittent 504 timeouts. "
                        "Latency increased 3x in the last hour. Logs show occasional Timeout errors."
    )
    print(f"\n  Severity  : {result_a['severity']}")
    print(f"  Escalated : {result_a['is_escalated']}")
    print("\n  Agent outputs:")
    print(f"\n  [TriageAgent]\n{result_a['triage_result']}")
    print(f"\n  [DiagnosticAgent]\n{result_a['diagnostic_result']}")
    print(f"\n  [ResolutionAgent]\n{result_a['resolution_runbook']}")
    print(f"\n  [QAAgent]\n{result_a['qa_result']}")

    # Test B: P1 incident (escalation path)
    print("\n── Test B: P1 Incident (escalation path) ──")
    result_b = workflow.run(
        incident_id   = f"INC-{uuid.uuid4().hex[:6].upper()}",
        incident_text = "PRODUCTION DOWN. PostgreSQL database completely unavailable. "
                        "OOMKilled events in logs. All 2000 users cannot access the system."
    )
    print(f"\n  Severity  : {result_b['severity']}")
    print(f"  Escalated : {result_b['is_escalated']}")
    print("\n  Agent outputs:")
    print(f"\n  [TriageAgent]\n{result_b['triage_result']}")
    print(f"\n  [DiagnosticAgent]\n{result_b['diagnostic_result']}")
    print(f"\n  [ResolutionAgent]\n{result_b['resolution_runbook']}")
    print(f"\n  [QAAgent]\n{result_b['qa_result']}")

    print("\n✅ STEP 4 COMPLETE — LangGraph workflow running both paths")