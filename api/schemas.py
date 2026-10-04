"""Validated request and response schemas for the local API."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PortfolioRequest(BaseModel):
    """Portfolio selection shared by calculation endpoints."""

    model_config = ConfigDict(extra="forbid")
    assets: list[str] = Field(min_length=1)
    weights: dict[str, float]
    initial_value: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def matching_unique_assets(self) -> "PortfolioRequest":
        if any(not asset.strip() for asset in self.assets):
            raise ValueError("asset names must be non-empty")
        if len(self.assets) != len(set(self.assets)):
            raise ValueError("duplicate assets are not allowed")
        if set(self.assets) != set(self.weights):
            raise ValueError("weights must match selected assets exactly")
        return self


class RiskRequest(PortfolioRequest):
    """Request for one loss-based VaR/ES model."""

    confidence_level: float = Field(gt=0, lt=1, allow_inf_nan=False)
    model: Literal["historical", "gaussian", "student_t", "monte_carlo", "garch"]
    n_simulations: int = Field(default=10_000, gt=0)
    random_seed: int = Field(default=2026, ge=0)


class StressRequest(PortfolioRequest):
    """Request for one named supported Phase 6 scenario."""

    scenario: Literal[
        "uniform_5",
        "uniform_10",
        "uniform_20",
        "asset_specific_adverse",
        "gfc_replay",
        "covid_replay",
    ]
    base_value: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    asset_shocks: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str


class RiskResponse(BaseModel):
    model: str
    confidence_level: float
    var: float
    expected_shortfall: float
    sample_size: int
    unit: str
    loss_convention: str
    model_parameters: dict[str, Any]


class PortfolioResponse(BaseModel):
    assets: list[str]
    weights: dict[str, float]
    initial_value: float
    final_value: float
    observations: int
    first_date: date
    last_date: date
    statistics: dict[str, float]


class StressResponse(BaseModel):
    scenario_name: str
    scenario_type: str
    base_value: float
    stressed_value: float
    pnl: float
    loss: float
    loss_percentage: float
    asset_impacts: list[dict[str, Any]]
    assumptions: list[str]
    start_date: date | None = None
    end_date: date | None = None


class ModelSnapshotResponse(BaseModel):
    available: bool
    period: str | None = None
    source: str | None = None
    models: list[dict] = Field(default_factory=list)
    champion: dict | None = None
