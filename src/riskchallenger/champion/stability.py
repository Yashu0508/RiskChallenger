"""Chronological subperiod analysis for VaR forecast stability."""

import numpy as np
import pandas as pd

from riskchallenger.backtesting.christoffersen import (
    christoffersen_independence_test,
    conditional_coverage_test,
)
from riskchallenger.backtesting.kupiec import kupiec_test
from riskchallenger.champion.results import PeriodMetrics


def evaluate_stability(
    forecasts: pd.DataFrame,
    confidence_level: float,
    significance_level: float,
) -> tuple[float, tuple[PeriodMetrics, ...]]:
    """Split chronological forecasts into equal-count early/middle/recent thirds.

    The stability score is the mean of three bounded components: subperiod
    coverage-test pass fraction, average normalized exception-rate accuracy,
    and normalized standard deviation of the three exception rates.
    """
    required = {"realized_loss", "var_forecast"}
    if not required.issubset(forecasts.columns):
        raise ValueError(
            "forecast frame requires realized_loss and var_forecast columns"
        )
    if not isinstance(forecasts.index, pd.DatetimeIndex):
        raise ValueError("forecast frame must have a DatetimeIndex")
    if forecasts.index.has_duplicates or not forecasts.index.is_monotonic_increasing:
        raise ValueError("forecast dates must be unique and chronological")
    if len(forecasts) < 90:
        raise ValueError("at least 90 forecasts are required (30 in each subperiod)")
    values = forecasts[["realized_loss", "var_forecast"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("subperiod forecasts contain missing or non-finite values")

    expected_rate = 1.0 - confidence_level
    chunks = np.array_split(np.arange(len(forecasts)), 3)
    names = ("early", "middle", "recent")
    metrics: list[PeriodMetrics] = []
    rates: list[float] = []
    passing: list[bool] = []
    normalized_accuracy: list[float] = []
    for name, positions in zip(names, chunks, strict=True):
        period = forecasts.iloc[positions]
        indicators = (
            period["realized_loss"].to_numpy() > period["var_forecast"].to_numpy()
        )
        count = int(indicators.sum())
        rate = count / len(period)
        kupiec = kupiec_test(indicators, confidence_level, significance_level)
        independence = christoffersen_independence_test(indicators, significance_level)
        conditional = conditional_coverage_test(
            indicators, confidence_level, significance_level
        )
        metrics.append(
            PeriodMetrics(
                period=name,
                start_date=period.index[0],
                end_date=period.index[-1],
                forecast_count=len(period),
                violations=count,
                violation_rate=float(rate),
                expected_rate=expected_rate,
                kupiec_p_value=kupiec.p_value,
                christoffersen_p_value=independence.p_value,
                conditional_coverage_p_value=conditional.p_value,
            )
        )
        rates.append(rate)
        passing.append(conditional.p_value >= significance_level)
        normalized_accuracy.append(
            max(0.0, 1.0 - abs(rate - expected_rate) / expected_rate)
        )

    pass_fraction = sum(passing) / len(passing)
    accuracy = float(np.mean(normalized_accuracy))
    normalized_variability = min(1.0, float(np.std(rates, ddof=0)) / expected_rate)
    variability_score = 1.0 - normalized_variability
    score = (pass_fraction + accuracy + variability_score) / 3.0
    return float(score), tuple(metrics)
