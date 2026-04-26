from django.shortcuts import render

# Create your views here.
# incidents/views.py
import requests
from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, get_object_or_404
from .models import Incident


FASTAPI_URL = getattr(settings, "FASTAPI_BASE_URL", "http://localhost:8080")


def dashboard(request):
    """List all incidents — fetches from local DB (synced from FastAPI)."""
    incidents = Incident.objects.all()
    # Severity counts for summary cards
    counts = {
        "P1": incidents.filter(severity="P1").count(),
        "P2": incidents.filter(severity="P2").count(),
        "total": incidents.count(),
        "escalated": incidents.filter(is_escalated=True).count(),
    }
    return render(request, "incidents/dashboard.html", {
        "incidents": incidents,
        "counts": counts,
    })


def submit_incident(request):
    """Form to submit a new incident → calls FastAPI → saves to local DB."""
    if request.method == "POST":
        title       = request.POST.get("title", "").strip()
        description = request.POST.get("description", "").strip()
        submitted_by = request.POST.get("submitted_by", "anonymous")

        if not title or not description:
            messages.error(request, "Title and description are required.")
            return render(request, "incidents/submit.html")

        try:
            # Call FastAPI — this runs the full LangGraph workflow
            response = requests.post(
                f"{FASTAPI_URL}/api/incidents",
                json={"title": title, "description": description, "submitted_by": submitted_by},
                timeout=120,   # LLM calls can take up to 60s
            )
            response.raise_for_status()
            data = response.json()

            # Save to local DB for fast reads
            Incident.objects.create(
                incident_id  = data["incident_id"],
                title        = title,
                description  = description,
                severity     = data["severity"],
                category     = data["category"],
                status       = data["status"],
                is_escalated = data["is_escalated"],
                triage       = data["triage"],
                diagnosis    = data["diagnosis"],
                runbook      = data["runbook"],
                qa_verdict   = data["qa_verdict"],
                submitted_by = submitted_by,
            )
            messages.success(request, f"Incident {data['incident_id']} processed successfully!")
            return redirect("incident_detail", incident_id=data["incident_id"])

        except requests.Timeout:
            messages.error(request, "The AI analysis timed out. Please try again.")
        except requests.RequestException as e:
            messages.error(request, f"Could not connect to AI backend: {e}")

    return render(request, "incidents/submit.html")


def incident_detail(request, incident_id):
    """Show full analysis for one incident."""
    incident = get_object_or_404(Incident, incident_id=incident_id)
    return render(request, "incidents/detail.html", {"incident": incident})


def code_review(request):
    """Submit code for AI review — calls FastAPI /api/code-review."""
    review_result = None
    if request.method == "POST":
        code     = request.POST.get("code", "")
        language = request.POST.get("language", "python")
        context  = request.POST.get("context", "")
        try:
            r = requests.post(
                f"{FASTAPI_URL}/api/code-review",
                json={"code": code, "language": language, "context": context},
                timeout=120,
            )
            r.raise_for_status()
            review_result = r.json()["review"]
        except requests.RequestException as e:
            messages.error(request, f"Code review failed: {e}")

    return render(request, "incidents/code_review.html", {"review_result": review_result})