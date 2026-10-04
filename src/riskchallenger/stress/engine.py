"""Portfolio stress scenario execution and attribution."""

import math

import numpy as np

from riskchallenger.logging import get_logger
from riskchallenger.portfolio.portfolio import Portfolio
from riskchallenger.stress.results import AssetImpact, StressResult
from riskchallenger.stress.scenarios import StressScenario

LOGGER = get_logger("stress.engine")


class StressTestEngine:
    """Apply validated scenario return vectors to an existing portfolio."""

    def run_scenario(
        self,
        portfolio: Portfolio,
        scenario: StressScenario,
        base_value: float | None = None,
    ) -> StressResult:
        """Calculate stressed value, P&L, loss, and asset-level attribution."""
        if not isinstance(portfolio, Portfolio):
            raise TypeError("portfolio must be a validated Phase 2 Portfolio")
        value = (
            portfolio.initial_value
            if base_value is None
            else self._validate_base_value(base_value)
        )
        try:
            resolved = scenario.resolve(portfolio.assets)
        except AttributeError as error:
            raise TypeError(
                "scenario must implement the StressScenario interface"
            ) from error

        shocks = dict(resolved.asset_returns)
        if len(shocks) != len(resolved.asset_returns):
            raise ValueError("resolved scenario contains duplicate assets")
        if set(shocks) != set(portfolio.assets):
            raise ValueError(
                "resolved scenario must provide exactly the portfolio assets"
            )

        impacts: list[AssetImpact] = []
        for asset, weight in zip(
            portfolio.assets, portfolio.weight_vector, strict=True
        ):
            shock = float(shocks[asset])
            if not math.isfinite(shock) or shock <= -1:
                raise ValueError(f"scenario produces an invalid price for {asset}")
            exposure = value * weight
            contribution = exposure * shock
            stressed_exposure = exposure * (1.0 + shock)
            if not np.isfinite([contribution, stressed_exposure]).all():
                raise ValueError(f"scenario produces a non-finite impact for {asset}")
            impacts.append(
                AssetImpact(
                    asset=asset,
                    weight=weight,
                    return_shock=shock,
                    base_exposure=float(exposure),
                    stressed_exposure=float(stressed_exposure),
                    pnl_contribution=float(contribution),
                    loss_contribution=float(-contribution),
                )
            )

        pnl = math.fsum(item.pnl_contribution for item in impacts)
        stressed_value = value + pnl
        if not math.isfinite(stressed_value) or stressed_value <= 0:
            raise ValueError(
                "scenario produces an invalid non-positive portfolio value"
            )
        loss = -pnl
        loss_percentage = 100.0 * loss / value
        if not math.isclose(
            math.fsum(item.pnl_contribution for item in impacts),
            pnl,
            rel_tol=1e-12,
            abs_tol=1e-8,
        ):
            raise ArithmeticError(
                "asset attribution does not reconcile to portfolio P&L"
            )
        result = StressResult(
            scenario_name=resolved.name,
            scenario_type=resolved.scenario_type,
            base_value=float(value),
            stressed_value=float(stressed_value),
            pnl=float(pnl),
            loss=float(loss),
            loss_percentage=float(loss_percentage),
            asset_impacts=tuple(impacts),
            assumptions=resolved.assumptions,
            start_date=resolved.start_date,
            end_date=resolved.end_date,
        )
        LOGGER.info(
            "Scenario %s completed: stressed_value=%.6f, loss=%.6f",
            result.scenario_name,
            result.stressed_value,
            result.loss,
        )
        return result

    @staticmethod
    def _validate_base_value(base_value: float) -> float:
        if isinstance(base_value, bool):
            raise ValueError("base_value must be a finite positive number")
        try:
            result = float(base_value)
        except (TypeError, ValueError) as error:
            raise ValueError("base_value must be a finite positive number") from error
        if not math.isfinite(result) or result <= 0:
            raise ValueError("base_value must be a finite positive number")
        return result
