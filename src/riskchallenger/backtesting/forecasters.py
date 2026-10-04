"""Adapters that expose existing Phase 3 VaR models to the backtester."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from riskchallenger.risk import (
    GARCHRiskModel,
    GaussianRiskModel,
    HistoricalRiskModel,
    MonteCarloRiskModel,
    RiskUnit,
    StudentTRiskModel,
)


@dataclass(frozen=True)
class ForecastTrainingData:
    """Strictly pre-forecast observations supplied to a VaR forecaster."""

    losses: np.ndarray
    returns: np.ndarray
    portfolio_value: float
    unit: RiskUnit
    refit_number: int
    dates: pd.DatetimeIndex | None = None
    forecast_date: pd.Timestamp | None = None


class VaRForecaster(Protocol):
    """Model adapter interface consumed by the generic backtesting engine."""

    model_name: str

    def fit(self, training_data: ForecastTrainingData) -> None: ...

    def forecast_var(self, confidence_level: float) -> float: ...


class HistoricalVaRForecaster:
    model_name = "historical"

    def __init__(self) -> None:
        self.model = HistoricalRiskModel()

    def fit(self, training_data: ForecastTrainingData) -> None:
        self.model = HistoricalRiskModel().fit(
            training_data.losses, unit=training_data.unit
        )

    def forecast_var(self, confidence_level: float) -> float:
        return self.model.calculate(confidence_level).var


class GaussianVaRForecaster:
    model_name = "gaussian"

    def __init__(self) -> None:
        self.model = GaussianRiskModel()

    def fit(self, training_data: ForecastTrainingData) -> None:
        self.model = GaussianRiskModel().fit(
            training_data.losses, unit=training_data.unit
        )

    def forecast_var(self, confidence_level: float) -> float:
        return self.model.calculate(confidence_level).var


class StudentTVaRForecaster:
    model_name = "student_t"

    def __init__(self) -> None:
        self.model = StudentTRiskModel()

    def fit(self, training_data: ForecastTrainingData) -> None:
        self.model = StudentTRiskModel().fit(
            training_data.losses, unit=training_data.unit
        )

    def forecast_var(self, confidence_level: float) -> float:
        return self.model.calculate(confidence_level).var


class MonteCarloVaRForecaster:
    model_name = "monte_carlo"

    def __init__(self, n_simulations: int = 10_000, random_seed: int = 2026) -> None:
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.model: MonteCarloRiskModel | None = None

    def fit(self, training_data: ForecastTrainingData) -> None:
        seed = self.random_seed + training_data.refit_number
        self.model = MonteCarloRiskModel(
            n_simulations=self.n_simulations, random_seed=seed
        ).fit(training_data.losses, unit=training_data.unit)

    def forecast_var(self, confidence_level: float) -> float:
        if self.model is None:
            raise RuntimeError("Monte Carlo forecaster must be fitted first")
        return self.model.calculate(confidence_level).var


class GARCHVaRForecaster:
    model_name = "garch_1_1_gaussian"
    refit_frequency = 1

    def __init__(self) -> None:
        self.model: GARCHRiskModel | None = None

    def fit(self, training_data: ForecastTrainingData) -> None:
        self.model = GARCHRiskModel().fit(
            training_data.returns,
            unit=training_data.unit,
            portfolio_value=training_data.portfolio_value,
        )

    def forecast_var(self, confidence_level: float) -> float:
        if self.model is None:
            raise RuntimeError("GARCH forecaster must be fitted first")
        return self.model.calculate(confidence_level).var
