"""Read-only routes for already validated model summaries."""

from fastapi import APIRouter

from api.services import load_summary

router = APIRouter(prefix="/models", tags=["model summaries"])


@router.get("/champion")
def champion_summary() -> dict:
    """Return persisted Phase 4/5 summaries; does not rerun backtests."""
    return load_summary("dashboard_phase4_summary.json")


@router.get("/ml-diagnostic")
def ml_diagnostic() -> dict:
    """Return persisted Phase 7 recent-period diagnostic."""
    return load_summary("dashboard_phase7_diagnostic.json")
