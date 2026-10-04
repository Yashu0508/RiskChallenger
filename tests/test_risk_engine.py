"""Tests for loss-based Historical, Gaussian, and Student-t risk models."""

import numpy as np
import pytest
from scipy.stats import norm, t

from riskchallenger.portfolio import Portfolio, calculate_portfolio, load_return_dataset
from riskchallenger.risk import (
    GaussianRiskModel,
    HistoricalRiskModel,
    RiskModel,
    RiskResult,
    RiskUnit,
    StudentTRiskModel,
)


def test_common_model_interface_and_explicit_unit() -> None:
    assert issubclass(HistoricalRiskModel, RiskModel)
    with pytest.raises(ValueError, match="unit must be explicitly"):
        HistoricalRiskModel().fit([1.0, 2.0], unit="currency")
    result = HistoricalRiskModel().fit([0.01, 0.02], unit="relative").calculate(0.5)
    assert isinstance(result, RiskResult)
    assert result.unit is RiskUnit.RELATIVE
    assert result.loss_convention.startswith("positive values are losses")


@pytest.mark.parametrize(
    "alpha", [0.0, 1.0, -0.95, 1.2, float("nan"), float("inf"), "bad"]
)
def test_invalid_confidence_rejected_by_every_model(alpha: object) -> None:
    for model in (HistoricalRiskModel(), GaussianRiskModel(), StudentTRiskModel()):
        model.fit([1.0, 2.0, 3.0], unit=RiskUnit.MONETARY)
        with pytest.raises(ValueError, match="0 < alpha < 1"):
            model.calculate(alpha)


@pytest.mark.parametrize(
    "losses", [[], [float("nan"), 1.0], [float("inf"), 1.0], [-float("inf"), 1.0]]
)
def test_empty_or_non_finite_losses_rejected(losses: list[float]) -> None:
    with pytest.raises(ValueError):
        HistoricalRiskModel().fit(losses, unit=RiskUnit.RELATIVE)


def test_non_numeric_and_multidimensional_losses_rejected() -> None:
    with pytest.raises(ValueError, match="numeric"):
        GaussianRiskModel().fit([1.0, "bad"], unit=RiskUnit.MONETARY)
    with pytest.raises(ValueError, match="one-dimensional"):
        HistoricalRiskModel().fit([[1.0, 2.0]], unit=RiskUnit.RELATIVE)


def test_model_minimum_sample_requirements() -> None:
    historical = HistoricalRiskModel().fit([100], unit="monetary")
    assert historical.calculate(0.95).var == 100
    with pytest.raises(ValueError, match="at least 2"):
        GaussianRiskModel().fit([100], unit="monetary")
    with pytest.raises(ValueError, match="at least 3"):
        StudentTRiskModel().fit([100, 101], unit="monetary")


def test_calculation_before_fit_fails() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        GaussianRiskModel().calculate(0.95)


def test_historical_var_uses_linear_empirical_quantile() -> None:
    losses = np.array([10.0, 20.0, 30.0, 40.0])
    result = HistoricalRiskModel().fit(losses, unit="monetary").calculate(0.75)
    # Linear 75th percentile lies one quarter of the way from 30 to 40.
    assert result.var == pytest.approx(32.5)
    assert result.model_parameters["quantile_method"].startswith("numpy linear")


def test_historical_es_fractional_tail_mass_and_ties() -> None:
    losses = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = HistoricalRiskModel().fit(losses, unit="relative").calculate(0.6)
    # 2 of 5 observations comprise the upper 40% tail; average (4 + 5) / 2.
    assert result.var == pytest.approx(3.4)
    assert result.expected_shortfall == pytest.approx(4.5)
    assert result.expected_shortfall >= result.var

    tied = (
        HistoricalRiskModel()
        .fit([1.0, 2.0, 3.0, 4.0, 4.0, 5.0], unit="relative")
        .calculate(0.5)
    )
    assert tied.expected_shortfall == pytest.approx(13.0 / 3.0)


def test_historical_small_sample_and_high_confidence() -> None:
    model = HistoricalRiskModel().fit([1.0, 2.0, 3.0, 4.0], unit="relative")
    assert model.calculate(0.999).var == pytest.approx(3.997)
    assert model.calculate(0.999).expected_shortfall == pytest.approx(4.0)
    one = HistoricalRiskModel().fit([7.0], unit="monetary").calculate(0.99)
    assert one.var == one.expected_shortfall == 7.0


def test_gaussian_var_and_es_match_independent_scipy_formulas() -> None:
    losses = np.array([-2.0, 1.0, 3.0, 4.0, 9.0])
    alpha = 0.95
    location = float(losses.mean())
    sample_scale = float(losses.std(ddof=1))
    z_alpha = float(norm.ppf(alpha))
    expected_var = location + sample_scale * z_alpha
    expected_es = location + sample_scale * norm.pdf(z_alpha) / (1 - alpha)

    result = GaussianRiskModel().fit(losses, unit="monetary").calculate(alpha)
    assert result.var == pytest.approx(expected_var)
    assert result.expected_shortfall == pytest.approx(expected_es)
    assert result.expected_shortfall >= result.var
    assert result.model_parameters["standard_deviation_ddof"] == 1


def test_gaussian_constant_losses_have_zero_variance() -> None:
    result = (
        GaussianRiskModel()
        .fit([100.0, 100.0, 100.0, 100.0], unit="monetary")
        .calculate(0.99)
    )
    assert result.var == pytest.approx(100.0)
    assert result.expected_shortfall == pytest.approx(100.0)
    assert result.model_parameters["scale_sample_standard_deviation"] == 0.0


@pytest.mark.parametrize("alpha", [0.99, 0.999])
def test_parametric_models_handle_high_confidence_levels(alpha: float) -> None:
    losses = [float(value) for value in [1, 3, 2, 4, 6, 5, 2, 8, 3, 7, 10, 4]]
    for model in (GaussianRiskModel(), StudentTRiskModel()):
        result = model.fit(losses, unit="relative").calculate(alpha)
        assert np.isfinite([result.var, result.expected_shortfall]).all()
        assert result.expected_shortfall >= result.var


def test_student_t_parameters_var_and_es_match_scipy() -> None:
    observations = t.rvs(
        df=5.0, loc=2.0, scale=3.0, size=500, random_state=np.random.default_rng(31415)
    )
    alpha = 0.95
    fitted_df, fitted_loc, fitted_scale = t.fit(observations)
    q_alpha = t.ppf(alpha, fitted_df)
    expected_var = fitted_loc + fitted_scale * q_alpha
    expected_es = fitted_loc + fitted_scale * (
        (fitted_df + q_alpha**2)
        / (fitted_df - 1)
        * t.pdf(q_alpha, fitted_df)
        / t.sf(q_alpha, fitted_df)
    )

    result = StudentTRiskModel().fit(observations, unit="relative").calculate(alpha)
    assert result.var == pytest.approx(expected_var, rel=1e-10)
    assert result.expected_shortfall == pytest.approx(expected_es, rel=1e-10)
    assert result.expected_shortfall >= result.var
    assert result.model_parameters["degrees_of_freedom"] == pytest.approx(fitted_df)
    assert "not standard deviation" in result.model_parameters["scale_definition"]


def test_student_t_constant_losses_are_handled_as_degenerate() -> None:
    result = (
        StudentTRiskModel().fit([3.0, 3.0, 3.0, 3.0], unit="relative").calculate(0.99)
    )
    assert result.var == result.expected_shortfall == 3.0
    assert result.model_parameters["degrees_of_freedom"] is None


def test_student_t_fit_rejects_infinite_es_degrees_of_freedom(monkeypatch) -> None:
    monkeypatch.setattr(
        "riskchallenger.risk.student_t.t.fit", lambda losses: (0.8, 0.0, 1.0)
    )
    with pytest.raises(ValueError, match="degrees_of_freedom > 1"):
        StudentTRiskModel().fit([1.0, 2.0, 3.0], unit="monetary").calculate(0.95)


def test_actual_phase_two_portfolio_loss_integration() -> None:
    returns = load_return_dataset()
    assets = ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"]
    portfolio = Portfolio(
        assets,
        {"HDFCBANK.NS": 0.50, "ICICIBANK.NS": 0.30, "RELIANCE.NS": 0.20},
        1_000_000,
    )
    portfolio_result = calculate_portfolio(portfolio, returns)
    losses = portfolio_result.loss
    assert len(losses) == 4184
    assert losses.index.equals(portfolio_result.portfolio_returns.index)

    results = [
        model.fit(losses, unit=RiskUnit.MONETARY).calculate(0.95)
        for model in (HistoricalRiskModel(), GaussianRiskModel(), StudentTRiskModel())
    ]
    assert [result.model_name for result in results] == [
        "historical",
        "gaussian",
        "student_t",
    ]
    assert all(result.sample_size == 4184 for result in results)
    assert all(result.unit is RiskUnit.MONETARY for result in results)
    assert all(
        np.isfinite([result.var, result.expected_shortfall]).all() for result in results
    )
    assert all(result.expected_shortfall >= result.var for result in results)
