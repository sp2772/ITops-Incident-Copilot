# IT Ops Copilot (LangGraph + FastAPI + Django UI)


View the website here: https://buzz-shortcake-defy.ngrok-free.dev/  (may or may not work)


An end-to-end **multi-agent IT incident management copilot** built as a learning project in 5 steps:

- **Step 1**: LLM client + token/cost logging (Google AI Studio via OpenAI-compatible API)
- **Step 2**: Deterministic LangChain tools (KnowlegeBase search, log analysis, runbook formatter)
- **Step 3**: ReAct-style agents built from those tools (with terminal tracing)
- **Step 4**: A LangGraph workflow that orchestrates the agents (branching + shared state)
- **Step 5**: A **FastAPI REST backend** + a **Django frontend** that calls it

---

## Project layout

Key files in `Langraph_FastAPI/`:

- `step1_llm_client.py`: LLM factory (`get_llm()`), raw LLM call helper (`direct_llm_call()`), and `CostTracker`
- `step2_lc_tools.py`: LangChain `@tool` functions (severity classification, KB search, log analysis, runbook formatting)
- `step3_lc_agents.py`: ReAct agents (Triage / Diagnostic / Resolution / QA) + trace logging of tool calls
- `step4_langgraph_workflow.py`: LangGraph `StateGraph` that runs agents in order and handles P1 escalation
- `step5_fastapi.py`: FastAPI API server exposing endpoints for incidents and code review
- `main.py`: “production-ish” entrypoint to start the API server (FastAPI runs on `:8080`)
- `requirements_linux.txt`: Python dependencies (pinned)
- `cost_log.json`: a local JSON cost log written by Step 1 (demo)

Django UI lives in:

- `copilot_ui/`
  - `manage.py`
  - `copilot_ui/settings.py`, `copilot_ui/urls.py`
  - `incidents/` app:
    - `models.py`: `Incident` model (stored in SQLite)
    - `views.py`: dashboard + submit form + detail page + code review form
    - `admin.py`: admin registration for `Incident`
    - `templates/incidents/*.html`: UI pages

---

## What the system does (high-level)

### Inputs
A user submits an incident:

- **title** (short)
- **description** (long / free text)

### Outputs
The system produces:

- **severity** (P1–P4)
- **category** (database/network/infrastructure/application/general)
- **root cause** hint (from patterns + reasoning)
- **resolution runbook** (markdown)
- **QA review** of the runbook

### Data flow (Django → FastAPI → agents → Django)
1. User submits the form in Django (`/submit/`)
2. Django calls FastAPI: `POST /api/incidents` (JSON body)
3. FastAPI runs the LangGraph workflow (Step 4), which calls agents (Step 3), which call tools (Step 2)
4. FastAPI returns a JSON response with all outputs
5. Django stores the result in its local SQLite DB (`Incident` model)
6. Dashboard and detail page read from Django DB (fast, no repeated LLM calls)

---

## Step-by-step explanation

### Step 1 — LLM Client + Cost logging (`step1_llm_client.py`)
- Uses `ChatOpenAI` but points it at **Google AI Studio** using:
  - `base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"`
  - `GOOGLE_API_KEY` from `.env` (create your own API keys in this file)
- `direct_llm_call(prompt)` does one LLM call and returns:
  - text
  - prompt tokens / completion tokens / total tokens
- `CostTracker` writes entries to `cost_log.json`

Why this exists:
- It isolates “how to talk to the model” and “how much it costs” from the rest of the app.

---

### Step 2 — Tools (`step2_lc_tools.py`)
Tools are deterministic functions exposed to the LLM:

- `classify_incident_severity(incident_description: str) -> dict`
- `search_knowledge_base(query: str) -> dict`
- `analyze_error_logs(log_text: str) -> dict`
- `format_runbook(resolution_steps_json: str) -> str`

Notes:
- KB search and log-pattern matching use **similarity scoring** (fuzzy matching) rather than exact `in` checks.
- `format_runbook` is hardened to handle common “LLM JSON formatting mistakes” and still produce valid markdown.

---

### Step 3 — Agents (`step3_lc_agents.py`)
Agents are built using `create_react_agent(...)` and small tool sets:

- **TriageAgent**: calls severity tool + KB tool and prints structured triage output
- **DiagnosticAgent**: calls log analyzer + KB tool and prints root cause + next step
- **ResolutionAgent**: calls KB tool and then `format_runbook`
  - includes terminal traces of tool calls
  - includes a single retry if it fails to call `format_runbook` (to reduce flakiness)
- **QAAgent**: no tools, just reviews the runbook

Tracing:
- Each agent invocation logs events like:
  - `LLM start/end`
  - `tool start/end`
  - tool inputs/outputs (truncated for readability)
  - and best-effort “LLM tool_calls” list

Why this exists:
- It makes LLM behavior observable (so you can debug: “did it call the tool?”).

---

### Step 4 — LangGraph workflow (`step4_langgraph_workflow.py`)
The workflow uses a typed shared state `IncidentState` and runs nodes in a graph:

- `triage_node`
- conditional route:
  - if P1 → `escalate_node` → `diagnostic_node`
  - else → `diagnostic_node`
- `resolution_node`
- `qa_node`
- END

The tests at the bottom run two scenarios and print **all agent outputs**.

---

### Step 5 — FastAPI server (`step5_fastapi.py`)
FastAPI exposes the workflow as a REST API:

- `GET /` → redirects to `/docs`
- `GET /health` → status JSON
- `POST /api/incidents` → runs full workflow and returns outputs
- `GET /api/incidents` → list incidents from FastAPI in-memory store
- `GET /api/incidents/{incident_id}` → single incident
- `POST /api/code-review` → single LLM call “code review”

Lifespan:
- The workflow is built once during startup (so it doesn’t rebuild on every request).

CORS:
- Configured so a Django frontend on port `8000` can call the API on port `8080`.

---

## Environment variables (`.env`)

The project expects (at minimum):

- `GOOGLE_API_KEY`: used by `step1_llm_client.py`
- `GCP_IP_ADDRESS`: used to build URLs when running on a VM (GCP)

Example:

```env
GOOGLE_API_KEY=your_key_here
GCP_IP_ADDRESS=136.145.YY.XXX
```

---

## Install & setup

### 1) Create and activate a virtualenv

```bash
python3 -m venv agents2
source agents2/bin/activate
```

### 2) Install dependencies

Recommended:

```bash
pip install -r requirements_linux.txt
```

---

## Run the system (local)

You typically run **two servers**:

### Terminal A — FastAPI (port 8080)

Option 1: run Step 5 directly:

```bash
cd Langraph_FastAPI
python3 step5_fastapi.py
```

Option 2: run the “entrypoint”:

```bash
cd Langraph_FastAPI
python3 main.py
```

Open:
- API docs: `http://localhost:8080/docs`
- Health: `http://localhost:8080/health`

### Terminal B — Django UI (port 8000)

```bash
cd Langraph_FastAPI/copilot_ui
python3 manage.py migrate
python3 manage.py runserver 8000
```

Open:
- UI: `http://localhost:8000`

---

## Run on GCP (public)

See `steps_instructions_to_follow.txt` for your exact GCP workflow.

Key concepts:

- Open firewall ports `8000` (Django) and `8080` (FastAPI)
- `ALLOWED_HOSTS` must include your external IP / ngrok domain
- `FASTAPI_BASE_URL` in Django `settings.py` must point to your FastAPI server (often the VM IP)

---

## Django UI details

Routes in the Django app (`copilot_ui/incidents/urls.py`):

- `/` → dashboard
- `/submit/` → submit an incident (POSTs to FastAPI, saves result to SQLite)
- `/incident/<id>/` → detail page
- `/code-review/` → code review page

Admin:
- Go to `/admin/`
- Incidents are visible because `incidents/admin.py` registers the `Incident` model

---

## Common troubleshooting

### 1) “Origin checking failed” / CSRF 403 on POST `/submit/`
Symptoms:

- `403 Forbidden (Origin checking failed ...)`

Fix:
- Add your HTTPS domain to `CSRF_TRUSTED_ORIGINS` in Django settings.py, e.g.

```python
CSRF_TRUSTED_ORIGINS = [
    "https://your-ngrok-subdomain.ngrok-free.dev",
    "https://*.ngrok-free.dev",
]
```

Also ensure:
- `ALLOWED_HOSTS` includes the domain

---

### 2) `http://localhost:8080/` returns `{"detail":"Not Found"}`
That’s normal if there is no `/` route. This project redirects `/` → `/docs`.

---

### 3) ResolutionAgent sometimes doesn’t call `format_runbook`
What it means:
- The model stopped after calling `search_knowledge_base`, or returned an empty final message.

What to do:
- Use the traces printed by `step3_lc_agents.py` / `step4_langgraph_workflow.py` to confirm:
  - whether `format_runbook` was called
  - what the tool input was
  - what it returned

---

### 4) `format_runbook` returns “## Resolution Runbook” + raw text
This happens when the input couldn’t be parsed cleanly as JSON.
The tool is hardened to recover from common mistakes, but if the model sends garbage, it will still fall back.

---

## Demo incidents you can try

- Title: `API gateway intermittent 504s`
  Description: `Intermittent 504s, latency up 3x, logs show Timeout errors after deployment.`

- Title: `PostgreSQL down`
  Description: `Production down, ConnectionRefused to postgres:5432, OOMKilled observed.`

---

## API reference (FastAPI)

Core endpoints:

- `GET /health`
- `POST /api/incidents`
- `GET /api/incidents`
- `GET /api/incidents/{incident_id}`
- `POST /api/code-review`

Use Swagger:
- `GET /docs`

---

## Notes on “what is FastAPI vs Django here?”

- **FastAPI**: runs the AI workflow and returns JSON (backend service)
- **Django**: serves HTML pages + stores incident history in SQLite (frontend UI + persistence)

This separation is intentional:
- FastAPI handles “compute + agents”
- Django handles “UI + database”

---

## License / Disclaimer

This is a demo/learning project. Do not use as-is for production incident response without adding:
- authentication/authorization
- rate limiting
- secret management
- persistent workflow storage
- observability (structured logging, tracing, metrics)
- safety guardrails for runbook steps

