"""Expanding/rolling out-of-sample VaR backtesting engine."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import numpy as np
import pandas as pd

from riskchallenger.backtesting.christoffersen import (
    christoffersen_independence_test,
    conditional_coverage_test,
)
from riskchallenger.backtesting.forecasters import (
    ForecastTrainingData,
    VaRForecaster,
)
from riskchallenger.backtesting.kupiec import kupiec_test
from riskchallenger.backtesting.results import (
    BacktestRunResult,
    ModelBacktestResult,
)
from riskchallenger.backtesting.validation import (
    parse_optional_date,
    validate_backtest_series,
    validate_confidence,
    validate_count,
    validate_significance_level,
)
from riskchallenger.backtesting.violations import (
    calculate_violation_statistics,
    violation_indicators,
)
from riskchallenger.logging import get_logger
from riskchallenger.portfolio.engine import PortfolioCalculationResult
from riskchallenger.risk.result import RiskUnit

LOGGER = get_logger("backtesting")


@dataclass(frozen=True)
class BacktestConfig:
    """Forecasting and statistical-test settings."""

    start_date: date | datetime | str | pd.Timestamp | None = None
    end_date: date | datetime | str | pd.Timestamp | None = None
    confidence_level: float = 0.95
    minimum_training_observations: int = 500
    refit_frequency: int = 1
    window_mode: str = "expanding"
    rolling_window: int | None = None
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "confidence_level", validate_confidence(self.confidence_level)
        )
        object.__setattr__(
            self,
            "minimum_training_observations",
            validate_count(
                self.minimum_training_observations, "minimum_training_observations"
            ),
        )
        object.__setattr__(
            self,
            "refit_frequency",
            validate_count(self.refit_frequency, "refit_frequency"),
        )
        object.__setattr__(
            self,
            "significance_level",
            validate_significance_level(self.significance_level),
        )
        if self.window_mode not in {"expanding", "rolling"}:
            raise ValueError("window_mode must be 'expanding' or 'rolling'")
        if self.rolling_window is not None:
            window = validate_count(self.rolling_window, "rolling_window")
            if window < self.minimum_training_observations:
                raise ValueError(
                    "rolling_window must be >= minimum_training_observations"
                )
            object.__setattr__(self, "rolling_window", window)
        start = parse_optional_date(self.start_date, "start_date")
        end = parse_optional_date(self.end_date, "end_date")
        if start is not None and end is not None and start > end:
            raise ValueError("start_date must not be after end_date")
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)

    @property
    def effective_rolling_window(self) -> int:
        """Fixed trailing sample length used by rolling mode."""
        return self.rolling_window or self.minimum_training_observations


class BacktestingEngine:
    """Generate strictly out-of-sample VaR forecasts and coverage tests."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        portfolio_result: PortfolioCalculationResult,
        forecasters: Sequence[VaRForecaster],
    ) -> BacktestRunResult:
        """Run each forecaster over a common date range and realized loss series."""
        if not forecasters:
            raise ValueError("at least one VaR forecaster is required")
        names = [forecaster.model_name for forecaster in forecasters]
        if len(names) != len(set(names)):
            raise ValueError("forecaster model_name values must be unique")

        losses = validate_backtest_series(portfolio_result.loss, "portfolio losses")
        returns = validate_backtest_series(
            portfolio_result.portfolio_returns, "portfolio returns"
        )
        values = validate_backtest_series(
            portfolio_result.portfolio_value, "portfolio values"
        )
        if not losses.index.equals(returns.index) or not losses.index.equals(
            values.index
        ):
            raise ValueError(
                "Phase 2 loss, return, and value series must be date-aligned"
            )

        start = self.config.start_date
        end = self.config.end_date
        date_mask = np.ones(len(losses), dtype=bool)
        if start is not None:
            date_mask &= losses.index >= start
        if end is not None:
            date_mask &= losses.index <= end
        eligible_forecast_positions = [
            position
            for position in range(
                self.config.minimum_training_observations, len(losses)
            )
            if date_mask[position]
        ]
        if not eligible_forecast_positions:
            raise ValueError(
                "date range contains no observations after the minimum training window"
            )

        results: dict[str, ModelBacktestResult] = {}
        for forecaster in forecasters:
            LOGGER.info("Starting out-of-sample backtest for %s", forecaster.model_name)
            model_started = time.perf_counter()
            model_refit_frequency = validate_count(
                getattr(forecaster, "refit_frequency", self.config.refit_frequency),
                f"{forecaster.model_name}.refit_frequency",
            )
            prepare_forecast = getattr(forecaster, "prepare_forecast", None)
            rows: list[dict[str, object]] = []
            fit_number = -1
            forecast_count = 0
            for position in eligible_forecast_positions:
                if self.config.window_mode == "rolling":
                    train_start = max(
                        0, position - self.config.effective_rolling_window
                    )
                else:
                    train_start = 0

                should_refit = forecast_count % model_refit_frequency == 0
                training = None
                if should_refit or callable(prepare_forecast):
                    training = ForecastTrainingData(
                        losses=losses.iloc[train_start:position].to_numpy(dtype=float),
                        returns=returns.iloc[train_start:position].to_numpy(
                            dtype=float
                        ),
                        portfolio_value=float(values.iloc[position - 1]),
                        unit=RiskUnit.MONETARY,
                        refit_number=fit_number + 1 if should_refit else fit_number,
                        dates=pd.DatetimeIndex(returns.index[train_start:position]),
                        forecast_date=pd.Timestamp(losses.index[position]),
                    )
                if should_refit:
                    fit_number += 1
                    assert training is not None
                    if training.losses.size < self.config.minimum_training_observations:
                        raise RuntimeError("constructed training window is too short")
                    forecaster.fit(training)
                if callable(prepare_forecast):
                    assert training is not None
                    prepare_forecast(training)

                var_forecast = float(
                    forecaster.forecast_var(self.config.confidence_level)
                )
                if not np.isfinite(var_forecast):
                    raise ValueError(
                        f"{forecaster.model_name} returned a non-finite VaR forecast"
                    )
                # Access realized loss only after the forecast has been formed.
                realized_loss = float(losses.iloc[position])
                rows.append(
                    {
                        "date": losses.index[position],
                        "realized_loss": realized_loss,
                        "var_forecast": var_forecast,
                        "violation": realized_loss > var_forecast,
                    }
                )
                forecast_metadata = getattr(forecaster, "forecast_metadata", None)
                if callable(forecast_metadata):
                    metadata = forecast_metadata()
                    if not isinstance(metadata, Mapping):
                        raise TypeError(
                            f"{forecaster.model_name} forecast_metadata must "
                            "return a mapping"
                        )
                    reserved = {
                        "date",
                        "realized_loss",
                        "var_forecast",
                        "violation",
                    }
                    if reserved.intersection(metadata):
                        raise ValueError(
                            "forecast metadata cannot overwrite core fields"
                        )
                    rows[-1].update(metadata)
                forecast_count += 1
                if forecast_count % 500 == 0:
                    LOGGER.info(
                        "%s generated %d/%d forecasts",
                        forecaster.model_name,
                        forecast_count,
                        len(eligible_forecast_positions),
                    )

            frame = pd.DataFrame(rows).set_index("date")
            frame.index = pd.DatetimeIndex(frame.index, name="date")
            violations = violation_indicators(
                frame["realized_loss"], frame["var_forecast"]
            )
            frame["violation"] = violations
            stats = calculate_violation_statistics(
                violations, self.config.confidence_level
            )
            kupiec = kupiec_test(
                violations.to_numpy(),
                self.config.confidence_level,
                self.config.significance_level,
            )
            christoffersen = christoffersen_independence_test(
                violations.to_numpy(), self.config.significance_level
            )
            conditional = conditional_coverage_test(
                violations.to_numpy(),
                self.config.confidence_level,
                self.config.significance_level,
            )
            results[forecaster.model_name] = ModelBacktestResult(
                model_name=forecaster.model_name,
                confidence_level=self.config.confidence_level,
                runtime_seconds=time.perf_counter() - model_started,
                forecasts=frame,
                violation_statistics=stats,
                kupiec=kupiec,
                christoffersen=christoffersen,
                conditional_coverage=conditional,
                refit_frequency=model_refit_frequency,
            )
            LOGGER.info(
                "Completed %s: %d forecasts, %d violations",
                forecaster.model_name,
                stats.number_of_forecasts,
                stats.violations,
            )
        return BacktestRunResult(config=self.config, model_results=results)


def run_backtest(
    portfolio_result: PortfolioCalculationResult,
    forecasters: Sequence[VaRForecaster],
    config: BacktestConfig | None = None,
) -> BacktestRunResult:
    """Convenience entry point for a configured VaR backtest."""
    return BacktestingEngine(config).run(portfolio_result, forecasters)
