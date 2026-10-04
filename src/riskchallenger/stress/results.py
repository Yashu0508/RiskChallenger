"""Typed stress outputs and per-asset P&L attribution."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AssetImpact:
    """One asset's fixed-weight shock and monetary contribution."""

    asset: str
    weight: float
    return_shock: float
    base_exposure: float
    stressed_exposure: float
    pnl_contribution: float
    loss_contribution: float


@dataclass(frozen=True)
class StressResult:
    """Portfolio valuation under one explicit deterministic scenario."""

    scenario_name: str
    scenario_type: str
    base_value: float
    stressed_value: float
    pnl: float
    loss: float
    loss_percentage: float
    asset_impacts: tuple[AssetImpact, ...]
    assumptions: tuple[str, ...]
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None

    @property
    def base_portfolio_value(self) -> float:
        """Explicitly named alias for the scenario's starting portfolio value."""
        return self.base_value

    @property
    def stressed_portfolio_value(self) -> float:
        """Explicitly named alias for the ending portfolio value."""
        return self.stressed_value

    @property
    def absolute_pnl(self) -> float:
        """Monetary P&L (signed), not its absolute magnitude."""
        return self.pnl

    @property
    def asset_impacts_frame(self) -> pd.DataFrame:
        """Return the asset attribution as a report-ready table."""
        return pd.DataFrame([impact.__dict__ for impact in self.asset_impacts])

    def summary(self) -> dict[str, float | str | pd.Timestamp | None]:
        """Return a compact scenario summary row."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "base_value": self.base_value,
            "base_portfolio_value": self.base_value,
            "stressed_value": self.stressed_value,
            "stressed_portfolio_value": self.stressed_value,
            "pnl": self.pnl,
            "absolute_pnl": self.pnl,
            "loss": self.loss,
            "loss_percentage": self.loss_percentage,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }
