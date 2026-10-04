"""Tests for out-of-sample VaR backtesting and coverage statistics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from riskchallenger.backtesting import (
    BacktestConfig,
    BacktestingEngine,
    ForecastTrainingData,
    GARCHVaRForecaster,
    GaussianVaRForecaster,
    HistoricalVaRForecaster,
    MonteCarloVaRForecaster,
    StudentTVaRForecaster,
    run_backtest,
)
from riskchallenger.backtesting.christoffersen import (
    christoffersen_independence_test,
    conditional_coverage_test,
)
from riskchallenger.backtesting.kupiec import kupiec_test
from riskchallenger.backtesting.violations import (
    calculate_violation_statistics,
    violation_indicators,
)
from riskchallenger.portfolio import Portfolio, calculate_portfolio, load_return_dataset


def make_phase2_result(returns: np.ndarray):
    dates = pd.date_range("2020-01-01", periods=len(returns), name="date")
    asset_returns = pd.DataFrame({"A": returns}, index=dates)
    portfolio = Portfolio(["A"], {"A": 1.0}, 1_000_000)
    return calculate_portfolio(portfolio, asset_returns)


class RecordingForecaster:
    model_name = "recording"

    def __init__(self):
        self.training_sets: list[np.ndarray] = []

    def fit(self, training_data: ForecastTrainingData) -> None:
        self.training_sets.append(training_data.losses.copy())

    def forecast_var(self, confidence_level: float) -> float:
        return float(np.mean(self.training_sets[-1]))


def test_strict_violation_definition_and_rate_statistics() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    realized = pd.Series([10.0, 11.0, 9.0], index=index)
    forecast = pd.Series([10.0, 10.0, 10.0], index=index)
    indicators = violation_indicators(realized, forecast)
    assert indicators.tolist() == [False, True, False]
    stats = calculate_violation_statistics(indicators, 0.95)
    assert stats.number_of_forecasts == 3
    assert stats.violations == 1
    assert stats.violation_rate == pytest.approx(1 / 3)
    assert stats.expected_violation_rate == pytest.approx(0.05)
    assert stats.rate_difference == pytest.approx(1 / 3 - 0.05)


def test_kupiec_expected_violation_count_has_zero_lr() -> None:
    violations = [True] * 5 + [False] * 95
    result = kupiec_test(violations, confidence_level=0.95)
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.degrees_of_freedom == 1
    assert result.reject_null is False


@pytest.mark.parametrize("violations", [[False] * 20, [True] * 20])
def test_kupiec_handles_zero_and_all_violations(violations: list[bool]) -> None:
    result = kupiec_test(violations, confidence_level=0.95)
    assert np.isfinite([result.statistic, result.p_value]).all()
    assert result.statistic >= 0
    assert 0 <= result.p_value <= 1


def test_christoffersen_transition_counts_for_constant_and_alternating_sequences() -> (
    None
):
    all_zero = christoffersen_independence_test([False] * 8)
    assert all_zero.transition_counts == {"N00": 7, "N01": 0, "N10": 0, "N11": 0}
    assert all_zero.statistic == pytest.approx(0.0)
    assert all_zero.p_value == pytest.approx(1.0)

    all_one = christoffersen_independence_test([True] * 5)
    assert all_one.transition_counts == {"N00": 0, "N01": 0, "N10": 0, "N11": 4}
    assert all_one.statistic == pytest.approx(0.0)

    alternating = christoffersen_independence_test(
        [False, True, False, True, False, True, False, True, False, True]
    )
    assert alternating.transition_counts == {
        "N00": 0,
        "N01": 5,
        "N10": 4,
        "N11": 0,
    }
    assert alternating.statistic > 0


def test_clustered_violations_have_positive_dependence_statistic() -> None:
    clustered = [False] * 20 + [True] * 6 + [False] * 20 + [True] * 6
    alternating = [False, True] * 12 + [False] * 28
    cluster_result = christoffersen_independence_test(clustered)
    alternating_result = christoffersen_independence_test(alternating)
    assert cluster_result.transition_counts["N11"] == 10
    assert cluster_result.statistic > 0
    assert alternating_result.statistic > 0


def test_conditional_coverage_is_sum_of_component_lrs() -> None:
    sequence = [False] * 12 + [True] * 3 + [False] * 10 + [True] * 4
    unconditional = kupiec_test(sequence, 0.95)
    independence = christoffersen_independence_test(sequence)
    combined = conditional_coverage_test(sequence, 0.95)
    assert combined.statistic == pytest.approx(
        unconditional.statistic + independence.statistic
    )
    assert combined.degrees_of_freedom == 2
    assert combined.p_value == pytest.approx(chi2.sf(combined.statistic, 2))


def test_coverage_tests_reject_invalid_or_too_short_sequences() -> None:
    with pytest.raises(ValueError, match="at least two"):
        christoffersen_independence_test([True])
    with pytest.raises(ValueError, match="non-empty"):
        kupiec_test([], 0.95)
    with pytest.raises(ValueError, match="exact 0/1"):
        kupiec_test([False, np.nan], 0.95)
    with pytest.raises(ValueError, match="0 < alpha < 1"):
        kupiec_test([False, True], 1.0)


def test_backtest_config_validates_windows_dates_and_refit() -> None:
    config = BacktestConfig(
        confidence_level=0.95,
        minimum_training_observations=5,
        refit_frequency=2,
        window_mode="rolling",
        rolling_window=8,
        start_date="2020-01-03",
        end_date="2020-01-10",
    )
    assert config.effective_rolling_window == 8
    with pytest.raises(ValueError, match="window_mode"):
        BacktestConfig(window_mode="invalid")
    with pytest.raises(ValueError, match="rolling_window"):
        BacktestConfig(
            minimum_training_observations=10, window_mode="rolling", rolling_window=5
        )
    with pytest.raises(ValueError, match="start_date"):
        BacktestConfig(start_date="2020-02-01", end_date="2020-01-01")
    with pytest.raises(ValueError, match="refit_frequency"):
        BacktestConfig(refit_frequency=0)


def test_expanding_window_and_refit_frequency_behavior() -> None:
    phase2 = make_phase2_result(np.linspace(-0.01, 0.01, 12))
    forecaster = RecordingForecaster()
    result = BacktestingEngine(
        BacktestConfig(minimum_training_observations=5, refit_frequency=3)
    ).run(phase2, [forecaster])
    model_result = result.model_results["recording"]
    assert len(model_result.forecasts) == 7
    assert [len(training) for training in forecaster.training_sets] == [5, 8, 11]
    assert model_result.forecasts.index.equals(phase2.loss.index[5:])


def test_rolling_window_is_fixed_and_date_bounds_are_applied() -> None:
    phase2 = make_phase2_result(np.linspace(-0.01, 0.01, 15))
    forecaster = RecordingForecaster()
    result = BacktestingEngine(
        BacktestConfig(
            start_date=phase2.loss.index[7],
            end_date=phase2.loss.index[9],
            minimum_training_observations=5,
            window_mode="rolling",
            rolling_window=5,
        )
    ).run(phase2, [forecaster])
    model_result = result.model_results["recording"]
    assert model_result.forecasts.index.equals(phase2.loss.index[7:10])
    assert [len(training) for training in forecaster.training_sets] == [5, 5, 5]
    assert np.array_equal(forecaster.training_sets[0], phase2.loss.iloc[2:7].to_numpy())


def test_no_look_ahead_when_final_realized_loss_changes() -> None:
    phase2 = make_phase2_result(np.linspace(-0.01, 0.01, 15))
    config = BacktestConfig(minimum_training_observations=5)
    original = BacktestingEngine(config).run(phase2, [GaussianVaRForecaster()])

    altered_losses = phase2.loss.copy()
    altered_losses.iloc[-1] += 10_000_000
    altered_phase2 = type(phase2)(
        portfolio_returns=phase2.portfolio_returns,
        portfolio_value=phase2.portfolio_value,
        pnl=phase2.pnl,
        loss=altered_losses,
        statistics=phase2.statistics,
    )
    altered = BacktestingEngine(config).run(altered_phase2, [GaussianVaRForecaster()])
    original_forecasts = original.model_results["gaussian"].forecasts
    altered_forecasts = altered.model_results["gaussian"].forecasts
    assert np.array_equal(
        original_forecasts["var_forecast"].to_numpy(),
        altered_forecasts["var_forecast"].to_numpy(),
    )
    assert (
        original_forecasts["var_forecast"].iloc[-1]
        == altered_forecasts["var_forecast"].iloc[-1]
    )
    assert (
        altered_forecasts["realized_loss"].iloc[-1]
        != original_forecasts["realized_loss"].iloc[-1]
    )


def test_backtest_requires_a_prior_training_window_and_unique_models() -> None:
    phase2 = make_phase2_result(np.linspace(-0.01, 0.01, 8))
    with pytest.raises(ValueError, match="no observations"):
        run_backtest(
            phase2,
            [RecordingForecaster()],
            BacktestConfig(minimum_training_observations=8),
        )
    with pytest.raises(ValueError, match="unique"):
        run_backtest(
            phase2,
            [RecordingForecaster(), RecordingForecaster()],
            BacktestConfig(minimum_training_observations=5),
        )


def test_actual_phase_two_integration_all_five_forecasters_short_test_slice() -> None:
    asset_returns = load_return_dataset()
    assets = ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"]
    portfolio = Portfolio(
        assets,
        {"HDFCBANK.NS": 0.50, "ICICIBANK.NS": 0.30, "RELIANCE.NS": 0.20},
        1_000_000,
    )
    phase2 = calculate_portfolio(portfolio, asset_returns)
    start = phase2.loss.index[500]
    end = phase2.loss.index[502]
    result = run_backtest(
        phase2,
        [
            HistoricalVaRForecaster(),
            GaussianVaRForecaster(),
            StudentTVaRForecaster(),
            MonteCarloVaRForecaster(n_simulations=2_000, random_seed=2026),
            GARCHVaRForecaster(),
        ],
        BacktestConfig(
            start_date=start,
            end_date=end,
            minimum_training_observations=500,
            refit_frequency=1,
            window_mode="expanding",
        ),
    )
    assert len(result.model_results) == 5
    assert {len(item.forecasts) for item in result.model_results.values()} == {3}
    assert result.comparison_table().shape[0] == 5
    assert (result.comparison_table()["number_of_forecasts"] == 3).all()
