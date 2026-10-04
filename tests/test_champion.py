"""Synthetic and integration tests for Champion–Challenger scoring."""

import numpy as np
import pandas as pd
import pytest

from riskchallenger.backtesting.christoffersen import (
    christoffersen_independence_test,
    conditional_coverage_test,
)
from riskchallenger.backtesting.kupiec import kupiec_test
from riskchallenger.backtesting.results import (
    BacktestRunResult,
    CoverageTestResult,
    ModelBacktestResult,
    ViolationStatistics,
)
from riskchallenger.champion import ChampionChallengerEvaluator, ScoreWeights
from riskchallenger.champion.scoring import (
    exception_rate_score,
    operational_score,
)
from riskchallenger.champion.stability import evaluate_stability


def make_result(
    name: str,
    violations: np.ndarray,
    runtime: float = 1.0,
    forecasts: pd.DataFrame | None = None,
) -> ModelBacktestResult:
    confidence = 0.95
    indicators = np.asarray(violations, dtype=bool)
    kupiec = kupiec_test(indicators, confidence)
    independence = christoffersen_independence_test(indicators)
    conditional = conditional_coverage_test(indicators, confidence)
    count = int(indicators.sum())
    rate = count / indicators.size
    stats = ViolationStatistics(
        number_of_forecasts=indicators.size,
        violations=count,
        violation_rate=rate,
        expected_violation_rate=0.05,
        rate_difference=rate - 0.05,
    )
    return ModelBacktestResult(
        model_name=name,
        confidence_level=confidence,
        runtime_seconds=runtime,
        forecasts=forecasts if forecasts is not None else pd.DataFrame(),
        violation_statistics=stats,
        kupiec=kupiec,
        christoffersen=independence,
        conditional_coverage=conditional,
    )


def make_run(*results: ModelBacktestResult) -> BacktestRunResult:
    return BacktestRunResult(
        config=object(), model_results={result.model_name: result for result in results}
    )


def random_indicators(seed: int = 42, n: int = 1_000, rate: float = 0.05):
    rng = np.random.default_rng(seed)
    values = np.zeros(n, dtype=bool)
    values[rng.choice(n, int(n * rate), replace=False)] = True
    return values


def test_exception_rate_score_and_operational_cost_formulas() -> None:
    assert exception_rate_score(0.05, 0.05) == pytest.approx(1.0)
    assert exception_rate_score(0.10, 0.05) == pytest.approx(0.0)
    assert exception_rate_score(0.04, 0.05) == pytest.approx(0.8)
    assert operational_score(60, 60) == pytest.approx(0.5)
    assert operational_score(0, 60) == pytest.approx(1.0)


def test_score_weights_are_configurable_and_validated() -> None:
    assert ScoreWeights(0.5, 0.2, 0.2, 0.1).coverage == pytest.approx(0.5)
    with pytest.raises(ValueError, match="sum to 1.0"):
        ScoreWeights(0.5, 0.2, 0.2, 0.2)
    with pytest.raises(ValueError, match="non-negative"):
        ScoreWeights(-0.1, 0.4, 0.4, 0.3)


def test_valid_model_eligible_and_invalid_model_ineligible() -> None:
    valid = make_result("valid", random_indicators())
    invalid = make_result("invalid", np.array([False] * 800 + [True] * 200))
    result = ChampionChallengerEvaluator().evaluate(make_run(valid, invalid))
    assert result.champion_name == "valid"
    assert result.ineligible_models == ("invalid",)
    assert result.evaluations[0].eligible
    assert not result.evaluations[1].eligible


def test_exception_accuracy_breaks_score_ranking_between_valid_models() -> None:
    near_target = make_result("near_target", random_indicators(seed=5, rate=0.05))
    farther = make_result("farther", random_indicators(seed=7, rate=0.07))
    result = ChampionChallengerEvaluator().evaluate(make_run(near_target, farther))
    assert result.champion_name == "near_target"
    assert (
        result.evaluations[0].exception_rate_score
        > result.evaluations[1].exception_rate_score
    )


def test_tied_models_use_deterministic_name_order() -> None:
    indicators = random_indicators(seed=19)
    zulu = make_result("zulu", indicators.copy(), runtime=4.0)
    alpha = make_result("alpha", indicators.copy(), runtime=4.0)
    first = ChampionChallengerEvaluator().evaluate(make_run(zulu, alpha))
    second = ChampionChallengerEvaluator().evaluate(make_run(alpha, zulu))
    assert first.champion_name == second.champion_name == "alpha"


def test_tie_breaks_coverage_then_exception_error_then_runtime() -> None:
    indicators = random_indicators(seed=23)
    base = make_result("z_base", indicators.copy(), runtime=4.0)
    fewer_passes = ModelBacktestResult(
        **{
            **make_result("a_fewer_passes", indicators.copy(), runtime=4.0).__dict__,
            "kupiec": CoverageTestResult(2.0, 0.01, 1, True),
        }
    )
    coverage_tie = ChampionChallengerEvaluator(
        weights=ScoreWeights(0.0, 0.5, 0.0, 0.5)
    ).evaluate(make_run(base, fewer_passes))
    assert coverage_tie.champion_name == "z_base"

    closer = make_result("z_closer", indicators.copy(), runtime=2.0)
    farther = make_result(
        "a_farther", random_indicators(seed=31, rate=0.06), runtime=2.0
    )
    accuracy_tie = ChampionChallengerEvaluator(
        weights=ScoreWeights(0.0, 0.0, 0.0, 1.0)
    ).evaluate(make_run(closer, farther))
    assert accuracy_tie.champion_name == "z_closer"

    quick = make_result("z_quick", indicators.copy(), runtime=1.0)
    slow = make_result("a_slow", indicators.copy(), runtime=5.0)
    runtime_tie = ChampionChallengerEvaluator(
        weights=ScoreWeights(0.0, 1.0, 0.0, 0.0)
    ).evaluate(make_run(quick, slow))
    assert runtime_tie.champion_name == "z_quick"


def test_operational_weight_changes_total_without_changing_coverage() -> None:
    indicators = random_indicators(seed=29)
    quick = make_result("quick", indicators.copy(), runtime=1.0)
    slow = make_result("slow", indicators.copy(), runtime=120.0)
    default = ChampionChallengerEvaluator().evaluate(make_run(quick, slow))
    operational_heavy = ChampionChallengerEvaluator(
        weights=ScoreWeights(0.1, 0.1, 0.1, 0.7)
    ).evaluate(make_run(quick, slow))
    default_scores = {item.model_name: item.total_score for item in default.evaluations}
    cost_scores = {
        item.model_name: item.total_score for item in operational_heavy.evaluations
    }
    assert cost_scores["quick"] - cost_scores["slow"] > (
        default_scores["quick"] - default_scores["slow"]
    )


def make_forecasts(indicators: np.ndarray, start: str = "2010-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(indicators), freq="D")
    realized = indicators.astype(float)
    return pd.DataFrame(
        {"realized_loss": realized, "var_forecast": np.full(len(indicators), 0.5)},
        index=index,
    )


def test_stability_subperiods_are_chronological_and_affect_score() -> None:
    rng = np.random.default_rng(53)
    stable = np.zeros(300, dtype=bool)
    for block in range(3):
        selected = rng.choice(
            np.arange(block * 100, (block + 1) * 100), 5, replace=False
        )
        stable[selected] = True
    unstable = np.zeros(300, dtype=bool)
    unstable[:30] = True
    stable_score, periods = evaluate_stability(make_forecasts(stable), 0.95, 0.05)
    unstable_score, _ = evaluate_stability(make_forecasts(unstable), 0.95, 0.05)
    assert [item.period for item in periods] == ["early", "middle", "recent"]
    assert [item.forecast_count for item in periods] == [100, 100, 100]
    assert periods[0].start_date < periods[1].start_date < periods[2].start_date
    assert stable_score > unstable_score


def test_stability_rejects_short_or_malformed_forecast_data() -> None:
    with pytest.raises(ValueError, match="at least 90"):
        evaluate_stability(make_forecasts(np.zeros(89, dtype=bool)), 0.95, 0.05)
    malformed = make_forecasts(np.zeros(90, dtype=bool)).drop(columns="var_forecast")
    with pytest.raises(ValueError, match="requires"):
        evaluate_stability(malformed, 0.95, 0.05)


def test_missing_forecast_rows_mark_stability_unavailable_and_renormalize() -> None:
    model = make_result("summary_only", random_indicators())
    result = ChampionChallengerEvaluator().evaluate(make_run(model))
    evaluation = result.evaluations[0]
    assert evaluation.stability_score is None
    assert "weights renormalized" in evaluation.reason
    assert result.ranking.iloc[0]["stability_score"] is None


def test_configuration_and_confidence_consistency_are_validated() -> None:
    with pytest.raises(ValueError, match="significance_level"):
        ChampionChallengerEvaluator(significance_level=1.0)
    with pytest.raises(ValueError, match="greater than 0"):
        ChampionChallengerEvaluator(operational_time_scale_seconds=0)
    model = make_result("model", random_indicators())
    changed = ModelBacktestResult(**{**model.__dict__, "confidence_level": 0.99})
    with pytest.raises(ValueError, match="expected violation rate"):
        ChampionChallengerEvaluator().evaluate(make_run(changed))


def test_actual_phase_four_summary_integration_does_not_rerun_backtest() -> None:
    summaries = [
        (
            "historical",
            480,
            0.130293,
            7.004899e-79,
            8.423885e-12,
            1.218573e-87,
            8.991241,
        ),
        ("gaussian", 383, 0.103963, 7.195070e-40, 1.552336e-06, 1.165225e-43, 7.008502),
        (
            "student_t",
            458,
            0.124321,
            4.113619e-69,
            6.215676e-12,
            4.966136e-78,
            3671.156786,
        ),
        (
            "monte_carlo",
            481,
            0.130565,
            2.450780e-79,
            1.136587e-11,
            5.734276e-88,
            5.138913,
        ),
        ("garch_1_1_gaussian", 170, 0.046145, 0.277054, 0.482645, 0.432929, 204.084499),
    ]
    results = []
    for name, count, rate, kupiec_p, independent_p, conditional_p, runtime in summaries:
        results.append(
            ModelBacktestResult(
                model_name=name,
                confidence_level=0.95,
                runtime_seconds=runtime,
                forecasts=pd.DataFrame(),
                violation_statistics=ViolationStatistics(
                    number_of_forecasts=3684,
                    violations=count,
                    violation_rate=count / 3684,
                    expected_violation_rate=0.05,
                    rate_difference=count / 3684 - 0.05,
                ),
                kupiec=CoverageTestResult(0.0, kupiec_p, 1, kupiec_p < 0.05),
                christoffersen=CoverageTestResult(
                    0.0, independent_p, 1, independent_p < 0.05
                ),
                conditional_coverage=CoverageTestResult(
                    0.0, conditional_p, 2, conditional_p < 0.05
                ),
            )
        )
    result = ChampionChallengerEvaluator().evaluate(make_run(*results))
    assert result.champion_name == "garch_1_1_gaussian"
    assert set(result.ineligible_models) == {
        "historical",
        "gaussian",
        "student_t",
        "monte_carlo",
    }
    assert result.evaluations[0].stability_score is None
    assert len(result.ranking) == 5
