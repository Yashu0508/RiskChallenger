"""Loss-based VaR/ES endpoints backed by Phase 3 implementations."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import RiskRequest, RiskResponse
from api.services import get_returns, risk_calculation

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("/var-es", response_model=RiskResponse)
def calculate_risk(
    request: RiskRequest, returns: pd.DataFrame = Depends(get_returns)
) -> dict:
    try:
        return risk_calculation(request, returns)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/compare", response_model=list[RiskResponse])
def compare_risk_models(
    request: RiskRequest, returns: pd.DataFrame = Depends(get_returns)
) -> list[dict]:
    """Calculate all supported one-period models without running backtests."""
    try:
        return [
            risk_calculation(request.model_copy(update={"model": model}), returns)
            for model in ("historical", "gaussian", "student_t", "monte_carlo", "garch")
        ]
    except (ValueError, TypeError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
