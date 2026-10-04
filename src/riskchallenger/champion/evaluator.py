"""Orchestration for transparent Champion–Challenger model selection."""

import math
from dataclasses import dataclass
from functools import cmp_to_key

from riskchallenger.backtesting.results import BacktestRunResult
from riskchallenger.champion.results import (
    ChampionChallengerResult,
    ModelEvaluation,
)
from riskchallenger.champion.scoring import (
    ScoreWeights,
    exception_rate_score,
    operational_score,
)
from riskchallenger.champion.stability import evaluate_stability
from riskchallenger.champion.validation import (
    validate_nonnegative,
    validate_probability,
)
from riskchallenger.logging import get_logger
from riskchallenger.risk.validation import validate_confidence_level

LOGGER = get_logger("champion.evaluator")


@dataclass(frozen=True)
class ChampionChallengerEvaluator:
    """Evaluate existing Phase 4 structured results without refitting models.

    ``operational_time_scale_seconds`` controls the identity-independent
    hyperbolic runtime score. If forecast rows were not retained, stability is
    reported as unavailable and its weight is excluded then renormalized.
    """

    weights: ScoreWeights = ScoreWeights()
    significance_level: float = 0.05
    conditional_coverage_required: bool = True
    operational_time_scale_seconds: float = 60.0
    tie_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "significance_level",
            validate_probability(self.significance_level, "significance_level"),
        )
        object.__setattr__(
            self,
            "operational_time_scale_seconds",
            validate_nonnegative(
                self.operational_time_scale_seconds,
                "operational_time_scale_seconds",
            ),
        )
        if self.operational_time_scale_seconds == 0:
            raise ValueError("operational_time_scale_seconds must be greater than 0")
        object.__setattr__(
            self,
            "tie_tolerance",
            validate_nonnegative(self.tie_tolerance, "tie_tolerance"),
        )
        if not isinstance(self.conditional_coverage_required, bool):
            raise ValueError("conditional_coverage_required must be a boolean")

    def evaluate(self, results: BacktestRunResult) -> ChampionChallengerResult:
        """Score and rank models in an existing Phase 4 result object."""
        if not isinstance(results, BacktestRunResult):
            raise TypeError("results must be a Phase 4 BacktestRunResult")
        if not results.model_results:
            raise ValueError("backtest results contain no models")
        alpha_values = {
            item.confidence_level for item in results.model_results.values()
        }
        if len(alpha_values) != 1:
            raise ValueError("all models must use the same confidence level")
        confidence_level = validate_confidence_level(next(iter(alpha_values)))
        expected_rate = 1.0 - confidence_level
        evaluations: list[ModelEvaluation] = []
        for name, item in results.model_results.items():
            self._validate_model_result(name, item, expected_rate)
            pvalues = (
                item.kupiec.p_value,
                item.christoffersen.p_value,
                item.conditional_coverage.p_value,
            )
            coverage_score = (
                sum(pvalue >= self.significance_level for pvalue in pvalues) / 3.0
            )
            exception_error = abs(
                item.violation_statistics.violation_rate - expected_rate
            )
            rate_score = exception_rate_score(
                item.violation_statistics.violation_rate, expected_rate
            )
            stability_score = None
            periods = ()
            if len(item.forecasts) >= 90:
                stability_score, periods = evaluate_stability(
                    item.forecasts,
                    confidence_level,
                    self.significance_level,
                )
            cost_score = operational_score(
                item.runtime_seconds, self.operational_time_scale_seconds
            )
            component_values = [
                (self.weights.coverage, coverage_score),
                (self.weights.exception_rate, rate_score),
                (self.weights.stability, stability_score),
                (self.weights.operational_cost, cost_score),
            ]
            available_weight = sum(
                weight for weight, score in component_values if score is not None
            )
            if available_weight <= 0:
                raise ValueError(
                    "at least one scoring component must have positive weight"
                )
            total = (
                sum(
                    weight * float(score)
                    for weight, score in component_values
                    if score is not None
                )
                / available_weight
            )
            eligible = (
                item.conditional_coverage.p_value >= self.significance_level
                if self.conditional_coverage_required
                else True
            )
            if eligible:
                reason = (
                    "Conditional-coverage test not rejected at "
                    f"{self.significance_level:g}; composite score {total:.6f}."
                )
            else:
                reason = (
                    "Ineligible: conditional-coverage test rejected "
                    f"(p={item.conditional_coverage.p_value:.6g} < "
                    f"{self.significance_level:g})."
                )
            if stability_score is None:
                reason += (
                    " Stability score unavailable (forecast rows absent or fewer "
                    "than 90); available weights renormalized."
                )
            evaluations.append(
                ModelEvaluation(
                    model_name=name,
                    eligible=eligible,
                    champion=False,
                    total_score=float(total),
                    coverage_score=float(coverage_score),
                    exception_rate_score=float(rate_score),
                    stability_score=stability_score,
                    operational_score=float(cost_score),
                    violation_rate=item.violation_statistics.violation_rate,
                    expected_violation_rate=expected_rate,
                    kupiec_p_value=item.kupiec.p_value,
                    christoffersen_p_value=item.christoffersen.p_value,
                    conditional_coverage_p_value=item.conditional_coverage.p_value,
                    runtime_seconds=item.runtime_seconds,
                    exception_rate_error=float(exception_error),
                    passing_coverage_tests=int(
                        sum(pvalue >= self.significance_level for pvalue in pvalues)
                    ),
                    subperiods=periods,
                    reason=reason,
                )
            )

        ranked = sorted(evaluations, key=cmp_to_key(self._compare))
        champion_name = next(
            (item.model_name for item in ranked if item.eligible), None
        )
        finalized = tuple(
            ModelEvaluation(
                **{
                    **item.__dict__,
                    "champion": item.model_name == champion_name,
                }
            )
            for item in ranked
        )
        challengers = tuple(
            item.model_name
            for item in finalized
            if item.eligible and item.model_name != champion_name
        )
        ineligible = tuple(item.model_name for item in finalized if not item.eligible)
        LOGGER.info(
            "Champion evaluation completed: champion=%s, eligible_challengers=%d, "
            "ineligible=%d",
            champion_name,
            len(challengers),
            len(ineligible),
        )
        return ChampionChallengerResult(
            champion_name=champion_name,
            challenger_names=challengers,
            ineligible_models=ineligible,
            evaluations=finalized,
            significance_level=self.significance_level,
        )

    @staticmethod
    def _validate_model_result(name: str, item: object, expected_rate: float) -> None:
        if name != item.model_name:
            raise ValueError("model result key must match model_name")
        if not math.isclose(
            item.violation_statistics.expected_violation_rate,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("model expected violation rate conflicts with confidence")
        if item.violation_statistics.number_of_forecasts <= 0:
            raise ValueError("model must contain at least one forecast")
        if not 0 <= item.violation_statistics.violation_rate <= 1:
            raise ValueError("model violation_rate must be between zero and one")
        if (
            not 0
            <= item.violation_statistics.violations
            <= (item.violation_statistics.number_of_forecasts)
        ):
            raise ValueError("model violation count is outside the forecast count")
        observed_rate = (
            item.violation_statistics.violations
            / item.violation_statistics.number_of_forecasts
        )
        if not math.isclose(
            item.violation_statistics.violation_rate,
            observed_rate,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("model violation_rate is inconsistent with its counts")
        if item.runtime_seconds < 0 or not math.isfinite(item.runtime_seconds):
            raise ValueError("model runtime_seconds must be finite and non-negative")
        for label, test in (
            ("Kupiec", item.kupiec),
            ("Christoffersen", item.christoffersen),
            ("conditional coverage", item.conditional_coverage),
        ):
            if not 0.0 <= test.p_value <= 1.0 or not math.isfinite(test.p_value):
                raise ValueError(f"{label} p-value must be in [0, 1]")

    def _compare(self, left: ModelEvaluation, right: ModelEvaluation) -> int:
        """Sort by eligibility, score, then documented deterministic tie breaks."""
        if left.eligible != right.eligible:
            return -1 if left.eligible else 1
        if abs(left.total_score - right.total_score) > self.tie_tolerance:
            return -1 if left.total_score > right.total_score else 1
        if left.passing_coverage_tests != right.passing_coverage_tests:
            return (
                -1 if left.passing_coverage_tests > right.passing_coverage_tests else 1
            )
        if not math.isclose(
            left.exception_rate_error,
            right.exception_rate_error,
            rel_tol=0.0,
            abs_tol=self.tie_tolerance,
        ):
            return -1 if left.exception_rate_error < right.exception_rate_error else 1
        if not math.isclose(
            left.runtime_seconds,
            right.runtime_seconds,
            rel_tol=0.0,
            abs_tol=self.tie_tolerance,
        ):
            return -1 if left.runtime_seconds < right.runtime_seconds else 1
        return (left.model_name > right.model_name) - (
            left.model_name < right.model_name
        )
