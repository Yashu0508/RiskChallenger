"""Explicit scenario stress-test endpoint."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import StressRequest, StressResponse
from api.services import get_prices, stress_calculation

router = APIRouter(prefix="/stress", tags=["stress"])


@router.post("/run", response_model=StressResponse)
def run_stress(
    request: StressRequest, prices: pd.DataFrame = Depends(get_prices)
) -> dict:
    try:
        return stress_calculation(request, prices)
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
