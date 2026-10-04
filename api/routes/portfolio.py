"""Portfolio analysis endpoint."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import PortfolioRequest, PortfolioResponse
from api.services import get_returns, portfolio_analysis

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/analyze", response_model=PortfolioResponse)
def analyze_portfolio(
    request: PortfolioRequest, returns: pd.DataFrame = Depends(get_returns)
) -> dict:
    try:
        return portfolio_analysis(request, returns)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
