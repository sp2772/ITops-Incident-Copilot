# step3_lc_agents.py
"""
STEP 3 — LangChain ReAct Agents
=================================
Teaches:
  - create_react_agent: builds Reason→Act→Observe loop automatically
  - Each agent gets a focused tool subset (not all tools)
  - SystemMessage shapes the agent's persona and output format
  - How to extract the final text from the message list

ReAct loop (what happens inside create_react_agent):
  1. LLM reads system prompt + user message
  2. LLM decides: call a tool OR give final answer
  3. If tool: tool executes, result appended to messages
  4. Back to step 2 until LLM outputs final answer

Run: python step3_lc_agents.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent

from step1_llm_client import get_llm, DEFAULT_MODEL
from step2_lc_tools import (
    TRIAGE_TOOLS,
    DIAGNOSTIC_TOOLS,
    RESOLUTION_TOOLS,
)


# ── Helper: extract final text from agent result ──────────────────────────── #
class TraceCallbackHandler(BaseCallbackHandler):
    """
    Prints a readable trace of agent/LLM/tool activity to the terminal.
    Attach per-agent so logs are prefixed with the agent name.
    """

    def __init__(self, agent_name: str, max_tool_output_chars: int = 4000):
        self.agent_name = agent_name
        self.max_tool_output_chars = max_tool_output_chars

    def _p(self, msg: str):
        print(f"[trace:{self.agent_name}] {msg}", flush=True)
        pass

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = (serialized or {}).get("name") or "chain"
        self._p(f"started {name}")

    def on_chain_end(self, outputs, **kwargs):
        self._p("finished chain")

    def on_llm_start(self, serialized, prompts, **kwargs):
        name = (serialized or {}).get("name") or "llm"
        self._p(f"LLM start ({name})")

    def on_llm_end(self, response, **kwargs):
        # Best-effort: print the model's text output (if any).
        try:
            # LangChain typically returns LLMResult with .generations[0][0].text
            gens = getattr(response, "generations", None)
            if gens and gens[0] and gens[0][0]:
                gen0 = gens[0][0]
                text = getattr(gen0, "text", None)
                if isinstance(text, str) and text.strip():
                    t = text.strip()
                    if len(t) > 800:
                        t = t[:800] + "…"
                    self._p(f"LLM output: {t}")

                # Chat models sometimes store message with tool_calls
                msg = getattr(gen0, "message", None)
                if msg is not None:
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        names = [tc.get("name") for tc in tool_calls if isinstance(tc, dict)]
                        self._p(f"LLM tool_calls: {names}")
        except Exception:
            pass

        self._p("LLM end")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = (serialized or {}).get("name") or "tool"
        inp = input_str if isinstance(input_str, str) else str(input_str)
        if len(inp) > 800:
            inp = inp[:800] + "…"
        self._p(f"tool start: {tool_name} input={inp}")

    def on_tool_end(self, output, **kwargs):
        out = output if isinstance(output, str) else str(output)
        if len(out) > self.max_tool_output_chars:
            out = out[: self.max_tool_output_chars] + "…"
        self._p(f"tool end: output={out}")

    def on_tool_error(self, error, **kwargs):
        self._p(f"tool error: {error}")


def _invoke_with_trace(agent, agent_name: str, messages):
    """
    Invoke a langgraph agent with tracing enabled.
    """
    cb = TraceCallbackHandler(agent_name=agent_name)
    return agent.invoke({"messages": messages}, config={"callbacks": [cb]})


def _message_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Some LC message types can store rich content as a list of parts.
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join([x for x in parts if x.strip()])
    return str(content)


def last_message_text(result: dict) -> str:
    """
    create_react_agent returns {'messages': [...]}.
    Some models may end with an empty AIMessage (e.g., tool-call only),
    so we scan backwards for the last non-empty content.
    """
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        txt = _message_content_to_text(getattr(m, "content", ""))
        if isinstance(txt, str) and txt.strip():
            return txt
    return ""


def last_ai_message_text(result: dict) -> str:
    """
    Returns the last non-empty AIMessage content only (never ToolMessages).
    This prevents accidentally "returning" tool outputs as the agent answer.
    """
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        if m.__class__.__name__ != "AIMessage":
            continue
        txt = _message_content_to_text(getattr(m, "content", ""))
        if isinstance(txt, str) and txt.strip():
            return txt
    return ""


def last_tool_output(result: dict, tool_name: str) -> str:
    """
    Fallback: return the most recent ToolMessage content, ideally emitted
    right after calling `tool_name`. (Works even if the final AIMessage is empty.)
    """
    msgs = result.get("messages", [])
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.__class__.__name__ == "ToolMessage":
            # If the immediately previous message is an AIMessage that called the tool,
            # it will usually have tool_calls with the tool name.
            prev = msgs[i - 1] if i - 1 >= 0 else None
            tool_calls = getattr(prev, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    if tc.get("name") == tool_name:
                        return _message_content_to_text(getattr(m, "content", ""))
            # Otherwise, keep scanning; we only want the requested tool's output.
            continue
    return ""


# ── Agent 1: Triage Agent ─────────────────────────────────────────────────── #
class TriageAgent:
    """
    Classifies incident severity, category, SLA, and escalation requirement.
    Uses: classify_incident_severity + search_knowledge_base
    Output: structured triage summary as text
    """

    SYSTEM = SystemMessage(content="""You are an IT Ops Triage Specialist.
When given an incident description:
1. ALWAYS call classify_incident_severity first.
2. ALWAYS call search_knowledge_base with relevant keywords.
3. Summarize in this format:
   SEVERITY: <P1/P2/P3/P4>
   CATEGORY: <category>
   SLA: <minutes> minutes to resolve
   ESCALATE: <Yes/No>
   KNOWN ISSUE: <Yes/No — title if yes>
   SUMMARY: <2-sentence triage summary>
Be concise. No extra commentary.""")

    def __init__(self, model: str = DEFAULT_MODEL):
        llm = get_llm(model=model, temperature=0.1)  # low temp for consistent classification
        self.agent = create_react_agent(llm, tools=TRIAGE_TOOLS)

    def run(self, incident_text: str) -> str:
        result = _invoke_with_trace(
            self.agent,
            "TriageAgent",
            [self.SYSTEM, HumanMessage(content=incident_text)],
        )
        return last_ai_message_text(result) or last_message_text(result)


# ── Agent 2: Diagnostic Agent ─────────────────────────────────────────────── #
class DiagnosticAgent:
    """
    Analyzes logs/stack traces to identify root cause.
    Uses: analyze_error_logs + search_knowledge_base
    Output: root cause explanation and affected components
    """

    SYSTEM = SystemMessage(content="""You are a Senior Site Reliability Engineer (SRE).
When given an incident description and/or log data:
1. Call analyze_error_logs if any log text is provided.
2. Call search_knowledge_base for the identified error patterns.
3. Respond with:
   ROOT CAUSE: <concise explanation>
   AFFECTED COMPONENTS: <comma-separated list>
   CONFIDENCE: <High/Medium/Low>
   NEXT DIAGNOSTIC STEP: <what to check next>
Keep your analysis under 150 words, short and concise.""")

    def __init__(self, model: str = DEFAULT_MODEL):
        llm = get_llm(model=model, temperature=0.2)
        self.agent = create_react_agent(llm, tools=DIAGNOSTIC_TOOLS)

    def run(self, incident_text: str) -> str:
        result = _invoke_with_trace(
            self.agent,
            "DiagnosticAgent",
            [self.SYSTEM, HumanMessage(content=incident_text)],
        )
        return last_ai_message_text(result) or last_message_text(result)


# ── Agent 3: Resolution Agent ─────────────────────────────────────────────── #
class ResolutionAgent:
    """
    Generates a step-by-step resolution runbook.
    Uses: search_knowledge_base + format_runbook
    Output: formatted markdown runbook
    """

    SYSTEM = SystemMessage(content="""You are an IT Runbook Author.
Given an incident summary and its root cause:
1. Call search_knowledge_base to find existing resolution steps if any.
2. ALWAYS call format_runbook exactly once after you have enough info.
3. Build a comprehensive resolution plan (5-8 steps).
4. Call format_runbook with a JSON string containing:
   {"title": "...", "severity": "...", "category": "...",
    "steps": ["step1", "step2", ...],
    "prevention": "...",
    "notes": "Include verification steps and rollback plan"}
Rules:
- The JSON you pass to format_runbook MUST be valid JSON (double quotes, no trailing commas).
- Include at least 2 explicit verification commands and a rollback plan in notes.
- If search_knowledge_base returns no articles, still proceed by writing reasonable steps from scratch.
5. Return ONLY the formatted runbook output from format_runbook.
Do not add any text before or after the runbook.""")

    def __init__(self, model: str = DEFAULT_MODEL):
        llm = get_llm(model=model, temperature=0.15)
        self.agent = create_react_agent(llm, tools=RESOLUTION_TOOLS)

    def run(self, incident_summary: str) -> str:
        result = _invoke_with_trace(
            self.agent,
            "ResolutionAgent",
            [self.SYSTEM, HumanMessage(content=incident_summary)],
        )
        # Preferred: the agent's final AI message (should be markdown runbook).
        # Important: do NOT treat ToolMessage content (e.g., KB JSON) as final output.
        text = last_ai_message_text(result).strip()
        if text:
            return text

        # Fallback #1: use the actual `format_runbook` tool output (not other tool outputs).
        tool_text = last_tool_output(result, "format_runbook").strip()
        if tool_text:
            return tool_text

        # Retry once with a stronger instruction and KB context if available.
        kb_tool_text = last_tool_output(result, "search_knowledge_base").strip()
        retry_instruction = (
            "You MUST call `format_runbook` now.\n"
            "Do not call any other tools.\n"
            "Pass a VALID JSON string (double quotes) with keys: "
            "title, severity, category, steps (5-8 items), prevention, notes.\n"
            "Return ONLY the `format_runbook` output.\n"
        )
        retry_context = incident_summary
        if kb_tool_text:
            retry_context += f"\n\nKnowledge base search result (use this to build steps):\n{kb_tool_text}"

        retry_result = _invoke_with_trace(
            self.agent,
            "ResolutionAgent-retry",
            [self.SYSTEM, HumanMessage(content=retry_instruction + "\n\n" + retry_context)],
        )

        retry_text = last_ai_message_text(retry_result).strip()
        if retry_text:
            return retry_text

        retry_tool_text = last_tool_output(retry_result, "format_runbook").strip()
        if retry_tool_text:
            return retry_tool_text

        return "ERROR: ResolutionAgent did not return a final runbook and did not call format_runbook successfully (even after retry). See trace outputs for review."


# ── Agent 4: QA Agent ─────────────────────────────────────────────────────── #
class QAAgent:
    """
    Reviews the generated runbook for completeness and safety.
    Uses: no tools — pure LLM reasoning (validates structure and logic)
    Output: APPROVED or NEEDS_REVISION with specific feedback
    """

    SYSTEM = SystemMessage(content="""You are a QA Engineer reviewing IT runbooks.
Evaluate the runbook for:
  ✓ Does it have clear numbered steps?
  ✓ Does it address the stated root cause?
  ✓ Are steps safe (no data-destroying commands without backups)?
  ✓ Does it include a prevention section?

Don't be too strict, be lenient with the runbook. As long as the runbook answers the above criteria, it is fine.

Respond EXACTLY in this format:
VERDICT: APPROVED | NEEDS_REVISION
SCORE: <1-10>
ISSUES: <bullet list of issues, or "None">
RECOMMENDATION: <one sentence>""")

    def __init__(self, model: str = DEFAULT_MODEL):
        # No tools — QA is pure LLM reasoning
        llm = get_llm(model=model, temperature=0.1)
        self.agent = create_react_agent(llm, tools=[])

    def run(self, runbook: str) -> str:
        prompt = f"Review this runbook:\n\n{runbook}"
        result = _invoke_with_trace(
            self.agent,
            "QAAgent",
            [self.SYSTEM, HumanMessage(content=prompt)],
        )
        return last_ai_message_text(result) or last_message_text(result)


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 55)
    print("STEP 3 TEST — LangChain ReAct Agents")
    print("=" * 55)

    sample_incident = (
        "Our PostgreSQL database is down. All application pods are throwing "
        "ConnectionRefused errors. Logs show OOMKilled events. "
        "Production is affected — 100% of users cannot login."
    )

    print("\n[Test 1] TriageAgent")
    triage = TriageAgent()
    triage_result = triage.run(sample_incident)
    print(triage_result)

    print("\n[Test 2] DiagnosticAgent")
    diag = DiagnosticAgent()
    diag_result = diag.run(sample_incident)
    print(diag_result)

    print("\n[Test 3] ResolutionAgent")
    resolver = ResolutionAgent()
    summary = f"Incident: {sample_incident}\nTriage: {triage_result}\nDiagnosis: {diag_result}"
    resolution_result = resolver.run(summary)
    print(resolution_result[:500], "...")

    print("\n[Test 4] QAAgent")
    qa = QAAgent()
    qa_result = qa.run(resolution_result)
    print(qa_result)

    print("\n✅ STEP 3 COMPLETE — All 4 agents working via ReAct loop")