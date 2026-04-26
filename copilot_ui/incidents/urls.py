# incidents/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("",                          views.dashboard,        name="dashboard"),
    path("submit/",                   views.submit_incident,  name="submit_incident"),
    path("incident/<str:incident_id>/", views.incident_detail, name="incident_detail"),
    path("code-review/",              views.code_review,      name="code_review"),
]