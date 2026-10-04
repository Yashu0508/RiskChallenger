"""Out-of-sample VaR forecasting and coverage tests."""

from riskchallenger.backtesting.engine import (
    BacktestConfig,
    BacktestingEngine,
    BacktestRunResult,
    run_backtest,
)
from riskchallenger.backtesting.forecasters import (
    ForecastTrainingData,
    GARCHVaRForecaster,
    GaussianVaRForecaster,
    HistoricalVaRForecaster,
    MonteCarloVaRForecaster,
    StudentTVaRForecaster,
    VaRForecaster,
)

__all__ = [
    "BacktestConfig",
    "BacktestRunResult",
    "BacktestingEngine",
    "ForecastTrainingData",
    "GARCHVaRForecaster",
    "GaussianVaRForecaster",
    "HistoricalVaRForecaster",
    "MonteCarloVaRForecaster",
    "StudentTVaRForecaster",
    "VaRForecaster",
    "run_backtest",
]
