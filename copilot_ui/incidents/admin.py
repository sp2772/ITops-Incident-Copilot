from django.contrib import admin

# Django admin only shows models you register here.
# Without this, you'll only see built-in apps like Users/Groups.

from .models import Incident


@admin.register(Incident)
class IncidentAdmin(admin.ModelAdmin):
    # Columns shown in the admin "change list" page
    list_display = (
        "incident_id",
        "severity",
        "category",
        "status",
        "is_escalated",
        "submitted_by",
        "created_at",
    )
    # Filters in the right sidebar
    list_filter = ("severity", "category", "status", "is_escalated", "created_at")
    # Search box (searches these fields)
    search_fields = ("incident_id", "title", "description", "submitted_by")
    # Default ordering (newest first)
    ordering = ("-created_at",)
    # Read-only fields in the detail view (optional)
    readonly_fields = ("created_at",)
