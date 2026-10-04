"""RiskChallenger local FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import models, portfolio, risk, stress
from riskchallenger import __version__

app = FastAPI(
    title="RiskChallenger API",
    description="Local API for portfolio, risk, stress, and validated summaries.",
    version=__version__,
)
app.include_router(portfolio.router)
app.include_router(risk.router)
app.include_router(stress.router)
app.include_router(models.router)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Lightweight process health check; no dataset loading or model fitting."""
    return {"status": "ok", "project": "RiskChallenger", "version": __version__}


@app.exception_handler(FileNotFoundError)
async def missing_project_data(_: Request, error: FileNotFoundError) -> JSONResponse:
    """Report unavailable local project data without exposing filesystem paths."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Required local project data or artifact is unavailable."},
    )


@app.exception_handler(ValueError)
async def invalid_project_data(_: Request, error: ValueError) -> JSONResponse:
    """Translate malformed local inputs/artifacts without leaking paths."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Project data or computation validation failed."},
    )
