"""Typed Champion–Challenger evaluation outputs."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PeriodMetrics:
    """VaR calibration statistics for one chronological subperiod."""

    period: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    forecast_count: int
    violations: int
    violation_rate: float
    expected_rate: float
    kupiec_p_value: float
    christoffersen_p_value: float
    conditional_coverage_p_value: float


@dataclass(frozen=True)
class ModelEvaluation:
    """Component scores and transparent decision details for one model."""

    model_name: str
    eligible: bool
    champion: bool
    total_score: float
    coverage_score: float
    exception_rate_score: float
    stability_score: float | None
    operational_score: float
    violation_rate: float
    expected_violation_rate: float
    kupiec_p_value: float
    christoffersen_p_value: float
    conditional_coverage_p_value: float
    runtime_seconds: float
    exception_rate_error: float
    passing_coverage_tests: int
    subperiods: tuple[PeriodMetrics, ...]
    reason: str


@dataclass(frozen=True)
class ChampionChallengerResult:
    """Ranked evaluations and selected model labels."""

    champion_name: str | None
    challenger_names: tuple[str, ...]
    ineligible_models: tuple[str, ...]
    evaluations: tuple[ModelEvaluation, ...]
    significance_level: float

    @property
    def ranking(self) -> pd.DataFrame:
        """Return a report-ready ranking table with raw test p-values."""
        rows = []
        for rank, item in enumerate(self.evaluations, start=1):
            rows.append(
                {
                    "rank": rank,
                    "model": item.model_name,
                    "eligible": item.eligible,
                    "coverage_score": item.coverage_score,
                    "exception_rate_score": item.exception_rate_score,
                    "stability_score": item.stability_score,
                    "operational_score": item.operational_score,
                    "total_score": item.total_score,
                    "violation_rate": item.violation_rate,
                    "expected_violation_rate": item.expected_violation_rate,
                    "kupiec_p_value": item.kupiec_p_value,
                    "christoffersen_p_value": item.christoffersen_p_value,
                    "conditional_coverage_p_value": (item.conditional_coverage_p_value),
                    "runtime_seconds": item.runtime_seconds,
                    "status": (
                        "Champion"
                        if item.champion
                        else "Challenger"
                        if item.eligible
                        else "Ineligible"
                    ),
                    "reason": item.reason,
                }
            )
        return pd.DataFrame(rows)
