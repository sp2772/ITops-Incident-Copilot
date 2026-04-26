# step2_lc_tools.py
"""
STEP 2 — LangChain Tools
==========================
Teaches:
  - @tool turns any Python function into a LangChain Tool
  - The docstring IS the tool description the LLM reads — write it well
  - Tools are pure / deterministic (no LLM inside), fast and unit-testable
  - In production replace bodies with real API calls:
      classify_incident_severity → ServiceNow API
      search_knowledge_base      → Confluence / Elasticsearch
      analyze_error_logs         → Splunk / Datadog API
      format_runbook             → Jira / Confluence create-page API

Run: python step2_lc_tools.py
"""

import json
import re
from difflib import SequenceMatcher
import ast
from langchain_core.tools import tool


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokenize(s: str) -> list[str]:
    s = _normalize_text(s)
    return [t for t in s.split(" ") if t]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sequence_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _fuzzy_ratio(a: str, b: str) -> float:
    """
    Returns [0,1] similarity score. Uses rapidfuzz if installed, otherwise stdlib.
    """
    a_n, b_n = _normalize_text(a), _normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0

    try:
        # Optional dependency; nice-to-have.
        from rapidfuzz.fuzz import token_set_ratio  # type: ignore

        return float(token_set_ratio(a_n, b_n)) / 100.0
    except Exception:
        a_t, b_t = set(_tokenize(a_n)), set(_tokenize(b_n))
        return max(_sequence_ratio(a_n, b_n), _jaccard(a_t, b_t))


def _best_haystack_similarity(needle: str, haystack: str) -> float:
    """
    Needle is usually short (pattern/key). Haystack can be long (query/log line).
    """
    n = _normalize_text(needle)
    h = _normalize_text(haystack)
    if not n and not h:
        return 1.0
    if not n or not h:
        return 0.0
    if n in h:
        return 1.0

    # Compare to full line, and to individual tokens for camelcase-ish patterns.
    tokens = _tokenize(h)
    token_best = max((_fuzzy_ratio(n, t) for t in tokens), default=0.0)
    return max(_fuzzy_ratio(n, h), token_best)


# ── Tool 1: Severity Classifier ───────────────────────────────────────────── #
@tool
def classify_incident_severity(incident_description: str) -> dict:
    """
    Classify the severity (P1-P4) and category of an IT incident.

    Call this FIRST on every new incident.
    P1=critical/full outage, P2=major/partial outage,
    P3=moderate/degraded performance, P4=minor/cosmetic.
    Returns severity, category, SLA response time in minutes,
    and whether escalation is required.
    """
    desc = incident_description.lower()

    # Severity: keyword matching in descending priority
    if any(kw in desc for kw in ["down", "outage", "unavailable", "crash", "production down"]):
        severity = "P1"
    elif any(kw in desc for kw in ["slow", "degraded", "high latency", "timeout", "partial"]):
        severity = "P2"
    elif any(kw in desc for kw in ["error", "warning", "failed", "intermittent"]):
        severity = "P3"
    else:
        severity = "P4"

    # Category: what part of the stack
    if any(kw in desc for kw in ["database", "db", "sql", "postgres", "mysql", "mongo", "redis"]):
        category = "database"
    elif any(kw in desc for kw in ["network", "dns", "vpn", "firewall", "connectivity"]):
        category = "network"
    elif any(kw in desc for kw in ["server", "cpu", "memory", "disk", "kubernetes", "pod", "node", "k8s"]):
        category = "infrastructure"
    elif any(kw in desc for kw in ["api", "service", "endpoint", "http", "rest", "microservice"]):
        category = "application"
    else:
        category = "general"

    return {
        "severity":            severity,
        "category":            category,
        "sla_minutes":         {"P1": 15, "P2": 60, "P3": 240, "P4": 1440}[severity],
        "requires_escalation": severity == "P1",
    }


# ── Tool 2: Knowledge Base Search ────────────────────────────────────────── #
# Simulated KB — replace with Elasticsearch / Confluence search API
_KB: dict[str, dict] = {
    "database connection": {
        "title": "DB Connection Pool Exhaustion",
        "steps": [
            "Check active connections: SELECT count(*) FROM pg_stat_activity;",
            "Kill idle connections older than 10 min",
            "Increase max_connections in postgresql.conf",
            "Restart PgBouncer connection pooler",
        ],
        "prevention": "Monitor pg_stat_activity; set pool_size limits in app config",
    },
    "high cpu": {
        "title": "High CPU / Node Overload",
        "steps": [
            "Run: top -c  to identify hot process",
            "Check runaway queries: SELECT * FROM pg_stat_activity WHERE state='active';",
            "Scale deployment: kubectl scale deployment <name> --replicas=<n>",
            "Enable HPA (Horizontal Pod Autoscaler)",
        ],
        "prevention": "Set CPU resource requests/limits in Kubernetes manifests",
    },
    "api timeout": {
        "title": "API Timeout / High Latency",
        "steps": [
            "Check upstream service health and response times",
            "Review recent deployments for regressions (git log --oneline -20)",
            "Check rate-limit and retry-after response headers",
            "Enable circuit breaker if downstream is flaky",
        ],
        "prevention": "Implement exponential backoff + jitter in all service clients",
    },
    "pod crash": {
        "title": "Kubernetes Pod CrashLoopBackOff",
        "steps": [
            "kubectl logs <pod> --previous",
            "kubectl describe pod <pod>  (look for OOMKilled, readiness failures)",
            "If OOMKilled: increase memory limit in deployment manifest",
            "Review liveness/readiness probe timeouts",
        ],
        "prevention": "Set resource requests AND limits; never omit limits",
    },
    "dns": {
        "title": "DNS Resolution Failure",
        "steps": [
            "nslookup <hostname>  inside the affected pod",
            "Check /etc/resolv.conf",
            "Restart CoreDNS: kubectl rollout restart -n kube-system deployment/coredns",
            "Review NetworkPolicy — may be blocking UDP 53",
        ],
        "prevention": "Monitor CoreDNS error rate in Prometheus",
    },
}


@tool
def search_knowledge_base(query: str) -> dict:
    """
    Search the internal knowledge base for known issues and their runbooks.

    Use this to find established resolution steps for similar past incidents.
    Returns matching articles with step-by-step resolution and prevention tips.
    Always search before generating new steps from scratch.
    """
    q = query or ""

    scored: list[tuple[float, str, dict]] = []
    for key, article in _KB.items():
        score = _best_haystack_similarity(key, q)
        if score >= 0.55:
            scored.append((score, key, article))

    scored.sort(key=lambda t: t[0], reverse=True)
    matches = [
        {**article, "matched_key": key, "similarity": round(score, 3)}
        for score, key, article in scored
    ]
    return {
        "found":    bool(matches),
        "count":    len(matches),
        "articles": matches[:2],  # top 2 matches
    }


# ── Tool 3: Error Log Analyzer ───────────────────────────────────────────── #
_PATTERNS = {
    "OOMKilled":            "Out of memory — pod killed by kernel; increase memory limit",
    "ConnectionRefused":    "Target service unreachable — verify it is running",
    "CrashLoopBackOff":     "Container crashing repeatedly — check logs with --previous",
    "Timeout":              "Operation timed out — check latency and downstream load",
    "ECONNRESET":           "TCP connection reset — check firewall / security group rules",
    "NullPointerException": "Null reference in application code — check recent deploys",
    "OutOfMemoryError":     "JVM heap exhausted — increase -Xmx or fix memory leak",
    "FATAL":                "Fatal application error — page on-call immediately",
}


@tool
def analyze_error_logs(log_text: str) -> dict:
    """
    Analyze raw error logs or stack traces to identify root cause patterns.

    Use this when the incident report includes log output or a stack trace.
    Extracts known error signatures and explains their likely cause.
    Provide the raw log text as a single string.
    """
    lines = log_text.strip().split("\n")
    found, error_lines = [], []

    for line in lines:
        for pattern, explanation in _PATTERNS.items():
            score = _best_haystack_similarity(pattern, line)
            if score >= 0.72:
                found.append(
                    {
                        "pattern": pattern,
                        "explanation": explanation,
                        "line": line.strip(),
                        "similarity": round(score, 3),
                    }
                )
        if any(lvl in line.upper() for lvl in ["ERROR", "FATAL", "CRITICAL"]):
            error_lines.append(line.strip())

    found.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
    return {
        "total_lines":      len(lines),
        "error_count":      len(error_lines),
        "identified_issues": found[:5],
        "root_cause_hint":  found[0]["explanation"] if found else "No known patterns — manual review needed",
    }


# ── Tool 4: Runbook Formatter ─────────────────────────────────────────────── #
@tool
def format_runbook(resolution_steps_json: str) -> str:
    """
    Format resolution steps into a clean markdown runbook for the incident ticket.

    Input MUST be a JSON string with keys:
      title (str), severity (str), category (str),
      steps (list of str), prevention (str, optional), notes (str, optional)
    Use this as the LAST step to produce the final runbook artifact.
    """
    def _coerce_to_dict(payload) -> dict | None:
        # If tool is called with an already-parsed dict (some models do this), accept it.
        if isinstance(payload, dict):
            return payload

        if not isinstance(payload, str):
            payload = str(payload)

        s = payload.strip()
        if not s:
            return None

        # 1) Strict JSON
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # 2) Extract probable JSON object substring if model added surrounding text
        if "{" in s and "}" in s:
            s2 = s[s.find("{") : s.rfind("}") + 1].strip()
        else:
            s2 = s

        # 3) Try JSON again on extracted substring
        try:
            obj = json.loads(s2)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # 4) Python-literal style dict (single quotes) sometimes appears — parse safely
        try:
            obj = ast.literal_eval(s2)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    data = _coerce_to_dict(resolution_steps_json)
    if not data:
        return f"## Resolution Runbook\n\n{resolution_steps_json}"

    lines = [
        f"# Runbook: {data.get('title', 'Incident Resolution')}",
        f"\n**Severity**: {data.get('severity', 'Unknown')} | "
        f"**Category**: {data.get('category', 'Unknown')}",
        "\n## Resolution Steps\n",
    ]
    for i, step in enumerate(data.get("steps", []), 1):
        # LLMs sometimes include numbering inside the step text ("1. ...")
        step_s = str(step).strip()
        step_s = re.sub(r"^\s*\d+\.\s*", "", step_s)
        lines.append(f"{i}. {step_s}")

    if data.get("prevention"):
        lines.append(f"\n## Prevention\n{data['prevention']}")
    if data.get("notes"):
        lines.append(f"\n## Notes\n{data['notes']}")

    return "\n".join(lines)


# Grouped exports for use in step3
TRIAGE_TOOLS     = [classify_incident_severity, search_knowledge_base]
DIAGNOSTIC_TOOLS = [analyze_error_logs, search_knowledge_base]
RESOLUTION_TOOLS = [search_knowledge_base, format_runbook]
ALL_TOOLS        = [classify_incident_severity, search_knowledge_base,
                    analyze_error_logs, format_runbook]


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 55)
    print("STEP 2 TEST — LangChain Tools")
    print("=" * 55)

    print("\n[Test 1] classify_incident_severity")
    r = classify_incident_severity.invoke("Production postgres database is down — all users locked out")
    print(f"  {r}")

    print("\n[Test 2] search_knowledge_base")
    r = search_knowledge_base.invoke("database connection pool exhausted")
    print(f"  found={r['found']}, articles={r['count']}")
    if r["articles"]:
        print(f"  First: {r['articles'][0]['title']}")

    print("\n[Test 3] analyze_error_logs")
    logs = "2024-01-15 10:23 ERROR ConnectionRefused to postgres:5432\n" \
           "2024-01-15 10:23 FATAL OOMKilled: exceeded memory limit\n"
    r = analyze_error_logs.invoke(logs)
    print(f"  Issues found : {len(r['identified_issues'])}")
    print(f"  Root cause   : {r['root_cause_hint']}")

    print("\n[Test 4] format_runbook")
    payload = json.dumps({
        "title": "DB Connection Failure", "severity": "P1", "category": "database",
        "steps": ["Check pg_stat_activity", "Restart PgBouncer", "Verify credentials"],
        "prevention": "Monitor pool exhaustion via Prometheus pg_exporter",
    })
    print(format_runbook.invoke(payload)[:300])

    print("\n✅ STEP 2 COMPLETE — All 4 tools working")