# step5_fastapi.py
# This file creates a small REST API using FastAPI.
# If you know Django: think of this file as a combination of:
# - `urls.py` (route mapping)
# - `views.py` (view functions that handle requests)
# - `serializers.py` (request/response validation) — via Pydantic models
# - plus a tiny in-memory "database" dict for demo purposes.
"""
STEP 5 — FastAPI REST API
==========================
Teaches:
  - FastAPI app setup with lifespan (startup/shutdown hooks)
  - Pydantic models validate request bodies automatically
  - Dependency injection via function parameters
  - In-memory store (swap for PostgreSQL in production)
  - CORS middleware so Django frontend can call this API
  - Background-friendly sync routes (FastAPI runs them in thread pool)

Endpoints:
  POST /api/incidents          → process new incident → run full workflow
  GET  /api/incidents          → list all incidents
  GET  /api/incidents/{id}     → get one incident with full results
  POST /api/code-review        → quick code review (single LLM call)
  GET  /health                 → health check

Run: python step5_fastapi.py
Test: curl -X POST http://localhost:8000/api/incidents \
        -H "Content-Type: application/json" \
        -d '{"title":"DB down","description":"PostgreSQL is unavailable"}'
"""
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv(), override=True)
GCP_IP_ADDRESS = os.getenv("GCP_IP_ADDRESS")

# --- Standard library imports (built-in Python modules) ---
import uuid, time  # uuid: generate unique IDs; time: timestamps (similar to Django's timezone.now usage)
from contextlib import asynccontextmanager  # helps us create startup/shutdown hooks for FastAPI
from typing import Optional  # typing helper: "Optional[str]" means "str or None"

# --- Third-party imports (installed packages) ---
from fastapi import FastAPI, HTTPException  # FastAPI app object + HTTP error helper (like Django's Http404/ValidationError style)
from fastapi.middleware.cors import CORSMiddleware  # CORS: allows browser frontend to call this API from another origin/port
from fastapi.responses import RedirectResponse  # simple HTTP redirects (e.g., / → /docs)
from pydantic import BaseModel  # Pydantic models validate/parse JSON (similar role to DRF serializers)
import uvicorn  # ASGI server (similar to "runserver", but production-ish)

# --- Local project imports (your code from earlier steps) ---
from step1_llm_client import direct_llm_call, DEFAULT_MODEL  # single LLM call helper + which model name is being used
from step4_langgraph_workflow import IncidentWorkflow  # multi-agent workflow graph that triages/diagnoses/resolves incidents


# ── Pydantic Models (request / response shapes) ───────────────────────────── #
class IncidentRequest(BaseModel):
    # This model defines the JSON body the client must send to POST /api/incidents.
    # In Django terms: similar to defining the expected POST payload in a DRF serializer.
    title:       str  # required field: incident title
    description: str  # required field: incident description
    submitted_by: Optional[str] = "anonymous"  # optional field with a default


class IncidentResponse(BaseModel):
    # This model defines the JSON shape we return from incident endpoints.
    # FastAPI uses it both for validation and to generate docs at /docs.
    incident_id:  str   # e.g. "INC-ABC12345"
    status:       str   # "resolved" or "under_review" (demo status)
    severity:     str   # "P1".."P4" produced by the triage agent
    category:     str   # "database"/"application"/etc.
    is_escalated: bool  # True if severity==P1
    triage:       str   # text output from TriageAgent
    diagnosis:    str   # text output from DiagnosticAgent
    runbook:      str   # markdown output from ResolutionAgent/format_runbook tool
    qa_verdict:   str   # text output from QAAgent
    created_at:   str   # ISO-ish timestamp string


class CodeReviewRequest(BaseModel):
    # Request body for POST /api/code-review.
    code:     str                    # the code to review
    language: Optional[str] = "python"  # language label used in prompt formatting
    context:  Optional[str] = ""        # any extra context (repo, constraints, etc.)


class CodeReviewResponse(BaseModel):
    # Response body for POST /api/code-review.
    review:   str  # review text from the LLM
    language: str  # echo back the language used


# ── In-memory store (replace with SQLAlchemy + PostgreSQL in production) ───── #
# This is a simple dict acting like a database table keyed by incident_id.
# In Django you'd have a model + database; here we keep it in memory (data resets on restart).
_incident_store: dict[str, dict] = {}


# ── Lifespan: runs at startup and shutdown ─────────────────────────────────── #
_workflow: Optional[IncidentWorkflow] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan runs startup code before the first request and shutdown code
    after the last. We build the LangGraph workflow once here — expensive
    to rebuild per request.
    """
    global _workflow  # tell Python we are writing to the module-level variable
    print("[API] Starting up — building LangGraph workflow...")  # startup log
    _workflow = IncidentWorkflow()  # build the graph once (so each request doesn't rebuild it)
    print("[API] Workflow ready ✅")  # startup finished
    yield  # everything before this runs at startup; everything after this runs at shutdown
    # Shutdown section: in real apps you'd close DB connections, flush telemetry, etc.
    print("[API] Shutting down")  # shutdown log


# ── App ───────────────────────────────────────────────────────────────────── #
app = FastAPI(
    title="IT Ops Copilot API",  # appears in OpenAPI docs
    description="Multi-agent AI system for IT incident management",  # appears in /docs
    version="1.0.0",  # API version string (purely informational)
    lifespan=lifespan,  # attach startup/shutdown hook
)

# Allow Django frontend (localhost:8000 default Django port) to call this API
app.add_middleware(
    CORSMiddleware,  # middleware that adds CORS headers to responses
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", f"http://{GCP_IP_ADDRESS}:8000","https://buzz-shortcake-defy.ngrok-free.dev","https://*.ngrok-free.dev", "*"],  # who can call us from browser JS
    allow_methods=["*"],  # which HTTP methods are allowed (GET/POST/etc.)
    allow_headers=["*"],  # which headers the browser is allowed to send
)


# ── Endpoints ─────────────────────────────────────────────────────────────── #
@app.get("/")
def root():
    """Redirect homepage → interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    """Simple liveness probe — used by Kubernetes / load balancers."""
    # This returns JSON (FastAPI auto-converts dict → JSON response).
    return {"status": "healthy", "model": DEFAULT_MODEL, "incidents": len(_incident_store)}


@app.post("/api/incidents", response_model=IncidentResponse)
def create_incident(req: IncidentRequest):
    """
    Process a new IT incident through the full multi-agent workflow.
    This is a synchronous endpoint — FastAPI runs it in a thread pool
    automatically, keeping the event loop free.
    """
    # `req` is already validated by Pydantic (IncidentRequest).
    # If JSON is missing required fields, FastAPI returns 422 automatically.
    incident_id   = f"INC-{uuid.uuid4().hex[:8].upper()}"  # generate a unique incident id
    full_text     = f"Title: {req.title}\n\nDescription: {req.description}"  # create a single text blob for agents

    # Run the LangGraph workflow (step4)
    # `_workflow` is created at startup in lifespan().
    # In a production app, you'd want a safety check here; we assume lifespan ran.
    result = _workflow.run(incident_id=incident_id, incident_text=full_text)

    # Build response and persist to in-memory store
    record = {
        "incident_id":  incident_id,  # primary key
        # crude status derivation from QA output (demo logic)
        "status":       "resolved" if "APPROVED" in result.get("qa_result", "") else "under_review",
        "severity":     result.get("severity", "P3"),  # default if missing
        "category":     result.get("category", "general"),  # default if missing
        "is_escalated": result.get("is_escalated", False),  # escalation flag from graph
        "triage":       result.get("triage_result", ""),  # triage agent text
        "diagnosis":    result.get("diagnostic_result", ""),  # diagnostic agent text
        "runbook":      result.get("resolution_runbook", ""),  # resolution agent markdown
        "qa_verdict":   result.get("qa_result", ""),  # QA agent evaluation
        "submitted_by": req.submitted_by,  # not exposed in response_model below (we strip it out)
        "created_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),  # timestamp string (UTC)
    }
    _incident_store[incident_id] = record  # store it in memory
    # Build the response object, excluding submitted_by to match IncidentResponse fields.
    return IncidentResponse(**{k: v for k, v in record.items() if k != "submitted_by"})


@app.get("/api/incidents")
def list_incidents():
    """Return all incidents — summary view for the Django dashboard."""
    # Build a smaller list (like a "list view") with only summary fields.
    summaries = [
        {
            "incident_id": v["incident_id"],
            "status":      v["status"],
            "severity":    v["severity"],
            "category":    v["category"],
            "created_at":  v["created_at"],
        }
        for v in _incident_store.values()
    ]
    # Sort newest first
    return sorted(summaries, key=lambda x: x["created_at"], reverse=True)


@app.get("/api/incidents/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str):
    """Return full detail for one incident — used by Django detail view."""
    record = _incident_store.get(incident_id)  # fetch from in-memory "db"
    if not record:
        # FastAPI's HTTPException is how you return errors with status codes.
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    # Return the full record (minus submitted_by) as IncidentResponse.
    return IncidentResponse(**{k: v for k, v in record.items() if k != "submitted_by"})


@app.post("/api/code-review", response_model=CodeReviewResponse)
def code_review(req: CodeReviewRequest):
    """
    Quick code review — single LLM call (no agent loop needed here).
    Checks for bugs, security issues, and best practices.
    """
    # Build the prompt we send to the LLM.
    prompt = f"""You are a senior {req.language} engineer doing a code review.
Review this code for: bugs, security issues, performance, and best practices.

Context: {req.context or 'No additional context'}

Code:
```{req.language}
{req.code}
```

Format your review as:
BUGS: <list or None>
SECURITY: <list or None>
PERFORMANCE: <list or None>
BEST_PRACTICES: <list or None>
VERDICT: PASS | NEEDS_CHANGES | FAIL
SUMMARY: <2-sentence overall assessment>"""

    # Call the model once (step1 helper returns a dict with 'text' + token usage).
    result = direct_llm_call(prompt)
    # Return the review in a typed response model (FastAPI turns it into JSON).
    return CodeReviewResponse(review=result["text"], language=req.language)


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    # This block runs only when you execute: `python step5_fastapi.py`
    # (It does NOT run when this module is imported.)
    print("=" * 55)  # cosmetic banner
    print("STEP 5 — FastAPI Server starting on http://localhost:8080")  # where server will listen
    print("Docs at: http://localhost:8080/docs")  # Swagger/OpenAPI docs (like DRF browsable API, but nicer)
    print("=" * 55)  # cosmetic banner
    # Port 8080 so Django can still run on 8000 without conflict.
    # uvicorn.run starts the ASGI server that hosts the FastAPI `app`.
    uvicorn.run("step5_fastapi:app", host="0.0.0.0", port=8080, reload=False)