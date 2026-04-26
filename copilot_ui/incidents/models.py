from django.db import models

# Create your models here.
# incidents/models.py
# Stores incident records locally so Django can serve them without calling FastAPI every time.

class Incident(models.Model):
    incident_id  = models.CharField(max_length=50, unique=True)
    title        = models.CharField(max_length=200)
    description  = models.TextField()
    severity     = models.CharField(max_length=5)          # P1-P4
    category     = models.CharField(max_length=50)
    status       = models.CharField(max_length=30)
    is_escalated = models.BooleanField(default=False)
    triage       = models.TextField(blank=True)
    diagnosis    = models.TextField(blank=True)
    runbook      = models.TextField(blank=True)
    qa_verdict   = models.TextField(blank=True)
    submitted_by = models.CharField(max_length=100, default="anonymous")
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.severity}] {self.incident_id} — {self.title}"

    @property
    def severity_color(self):
        """CSS color class for severity badge — used in templates."""
        return {"P1": "danger", "P2": "warning", "P3": "info", "P4": "secondary"}.get(self.severity, "secondary")