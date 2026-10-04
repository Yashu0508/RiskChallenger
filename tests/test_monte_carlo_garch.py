"""Tests for the Phase 3B Monte Carlo and GARCH risk models."""

import warnings

import numpy as np
import pytest
from arch.utility.exceptions import ConvergenceWarning
from scipy.stats import norm

from riskchallenger.portfolio import Portfolio, calculate_portfolio, load_return_dataset
from riskchallenger.risk import (
    GARCHRiskModel,
    GaussianRiskModel,
    HistoricalRiskModel,
    MonteCarloRiskModel,
    RiskUnit,
    StudentTRiskModel,
)


def make_garch_returns(
    n_observations: int = 1200,
    seed: int = 8675309,
) -> np.ndarray:
    """Generate a reproducible Gaussian GARCH(1,1) return process."""
    rng = np.random.default_rng(seed)
    omega = 0.000002
    alpha = 0.08
    beta = 0.90
    variance = omega / (1 - alpha - beta)
    residuals = np.empty(n_observations)
    returns = np.empty(n_observations)
    for index in range(n_observations):
        residuals[index] = np.sqrt(variance) * rng.standard_normal()
        returns[index] = 0.0002 + residuals[index]
        variance = omega + alpha * residuals[index] ** 2 + beta * variance
    return returns


def test_monte_carlo_is_reproducible_for_same_seed() -> None:
    losses = np.linspace(-10.0, 50.0, 100)
    first = (
        MonteCarloRiskModel(n_simulations=20_000, random_seed=24)
        .fit(losses, unit="monetary")
        .calculate(0.95)
    )
    second = (
        MonteCarloRiskModel(n_simulations=20_000, random_seed=24)
        .fit(losses, unit="monetary")
        .calculate(0.95)
    )
    assert first.var == second.var
    assert first.expected_shortfall == second.expected_shortfall
    assert first.sample_size == 20_000
    assert first.model_parameters["random_seed"] == 24
    assert first.model_parameters["n_simulations"] == 20_000


def test_monte_carlo_seed_sensitivity() -> None:
    losses = np.arange(1000, dtype=float)
    first = (
        MonteCarloRiskModel(n_simulations=500, random_seed=1)
        .fit(losses, unit="relative")
        .calculate(0.95)
    )
    second = (
        MonteCarloRiskModel(n_simulations=500, random_seed=2)
        .fit(losses, unit="relative")
        .calculate(0.95)
    )
    assert (first.var, first.expected_shortfall) != (
        second.var,
        second.expected_shortfall,
    )


def test_monte_carlo_estimates_match_independent_bootstrap_sample() -> None:
    losses = np.linspace(-5.0, 45.0, 80)
    simulations = 15_000
    seed = 991
    alpha = 0.975
    simulated = np.random.default_rng(seed).choice(
        losses, size=simulations, replace=True
    )
    expected = HistoricalRiskModel().fit(simulated, unit="relative").calculate(alpha)
    observed = (
        MonteCarloRiskModel(n_simulations=simulations, random_seed=seed)
        .fit(losses, unit="relative")
        .calculate(alpha)
    )
    assert observed.var == expected.var
    assert observed.expected_shortfall == expected.expected_shortfall
    assert observed.expected_shortfall >= observed.var


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_simulations": 0},
        {"n_simulations": -1},
        {"n_simulations": 2.5},
        {"n_simulations": True},
        {"random_seed": -1},
        {"random_seed": 1.5},
        {"random_seed": None},
    ],
)
def test_monte_carlo_constructor_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MonteCarloRiskModel(**kwargs)


def test_monte_carlo_rejects_insufficient_historical_sample() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        MonteCarloRiskModel(n_simulations=100, random_seed=0).fit(
            np.arange(19, dtype=float), unit="relative"
        )


def test_monte_carlo_rejects_invalid_confidence_and_observations() -> None:
    model = MonteCarloRiskModel(n_simulations=100, random_seed=0).fit(
        np.arange(20, dtype=float), unit="relative"
    )
    with pytest.raises(ValueError, match="0 < alpha < 1"):
        model.calculate(1.0)
    with pytest.raises(ValueError, match="finite"):
        MonteCarloRiskModel(n_simulations=100, random_seed=0).fit(
            [*range(19), np.nan], unit="relative"
        )


def test_garch_fits_synthetic_series_and_forecasts_one_step() -> None:
    returns = make_garch_returns()
    result = GARCHRiskModel().fit(returns, unit="relative").calculate(0.95)
    assert result.model_name == "garch_1_1_gaussian"
    assert result.sample_size == len(returns)
    assert np.isfinite(
        [
            result.var,
            result.expected_shortfall,
            result.model_parameters["forecast_volatility"],
        ]
    ).all()
    assert result.model_parameters["innovation_distribution"] == "Gaussian"
    assert result.model_parameters["garch_specification"] == "GARCH(1,1)"
    assert result.model_parameters["convergence_flag"] == 0
    assert result.model_parameters["omega"] > 0
    assert result.model_parameters["alpha_1"] >= 0
    assert result.model_parameters["beta_1"] >= 0
    assert result.expected_shortfall >= result.var
    extreme = GARCHRiskModel().fit(returns, unit="relative").calculate(0.999)
    assert np.isfinite([extreme.var, extreme.expected_shortfall]).all()


def test_garch_independent_one_step_and_risk_formula_check() -> None:
    returns = make_garch_returns(n_observations=1500, seed=12)
    model = GARCHRiskModel().fit(returns, unit="relative")
    result = model.calculate(0.95)
    params = result.model_parameters
    z = float(norm.ppf(0.95))
    sigma = float(params["forecast_volatility"])
    mean = float(params["mean_return"])
    independent_var = -mean + sigma * z
    independent_es = -mean + sigma * float(norm.pdf(z)) / 0.05
    assert result.var == pytest.approx(independent_var, rel=1e-12)
    assert result.expected_shortfall == pytest.approx(independent_es, rel=1e-12)


def test_garch_monetary_conversion_requires_explicit_value() -> None:
    returns = make_garch_returns()
    relative = GARCHRiskModel().fit(returns, unit="relative").calculate(0.95)
    with pytest.raises(ValueError, match="explicitly supplied"):
        GARCHRiskModel().fit(returns, unit="monetary")
    with pytest.raises(ValueError, match="finite and positive"):
        GARCHRiskModel().fit(returns, unit="monetary", portfolio_value=float("inf"))
    with pytest.raises(ValueError, match="must be omitted"):
        GARCHRiskModel().fit(returns, unit="relative", portfolio_value=100)

    value = 1_000_000
    monetary = (
        GARCHRiskModel()
        .fit(returns, unit="monetary", portfolio_value=value)
        .calculate(0.95)
    )
    assert monetary.var == pytest.approx(relative.var * value)
    assert monetary.expected_shortfall == pytest.approx(
        relative.expected_shortfall * value
    )


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_garch_rejects_invalid_portfolio_value(value: float) -> None:
    with pytest.raises(ValueError, match="portfolio_value"):
        GARCHRiskModel().fit(
            make_garch_returns(), unit="monetary", portfolio_value=value
        )


def test_garch_rejects_bad_confidence_and_insufficient_or_invalid_returns() -> None:
    returns = make_garch_returns()
    model = GARCHRiskModel().fit(returns, unit="relative")
    with pytest.raises(ValueError, match="0 < alpha < 1"):
        model.calculate(0.0)
    with pytest.raises(ValueError, match="at least 100"):
        GARCHRiskModel().fit(np.arange(99, dtype=float), unit="relative")
    with pytest.raises(ValueError, match="unit must be explicitly"):
        GARCHRiskModel().fit(returns, unit="unknown")
    for invalid in (np.nan, np.inf, -np.inf):
        bad_returns = make_garch_returns()
        bad_returns[30] = invalid
        with pytest.raises(ValueError, match="finite"):
            GARCHRiskModel().fit(bad_returns, unit="relative")


def test_garch_constant_return_series_uses_degenerate_limit() -> None:
    returns = np.full(100, 0.01)
    relative = GARCHRiskModel().fit(returns, unit="relative").calculate(0.95)
    assert relative.var == pytest.approx(-0.01)
    assert relative.expected_shortfall == pytest.approx(-0.01)
    assert relative.model_parameters["forecast_volatility"] == 0.0

    monetary = (
        GARCHRiskModel()
        .fit(returns, unit="monetary", portfolio_value=1_000)
        .calculate(0.95)
    )
    assert monetary.var == pytest.approx(-10.0)


def test_garch_convergence_warning_is_not_silently_accepted(monkeypatch) -> None:
    class FakeResult:
        convergence_flag = 0

    class FakeModel:
        def fit(self, **kwargs):
            warnings.warn("test convergence warning", ConvergenceWarning)
            return FakeResult()

    monkeypatch.setattr(
        "riskchallenger.risk.garch.arch_model", lambda *args, **kwargs: FakeModel()
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        GARCHRiskModel().fit(make_garch_returns(), unit="relative")


def test_actual_phase_two_portfolio_runs_all_five_risk_models() -> None:
    returns = load_return_dataset()
    portfolio = Portfolio(
        ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"],
        {"HDFCBANK.NS": 0.50, "ICICIBANK.NS": 0.30, "RELIANCE.NS": 0.20},
        1_000_000,
    )
    portfolio_result = calculate_portfolio(portfolio, returns)
    losses = portfolio_result.loss
    historical = HistoricalRiskModel().fit(losses, unit="monetary").calculate(0.95)
    gaussian = GaussianRiskModel().fit(losses, unit="monetary").calculate(0.95)
    student_t = StudentTRiskModel().fit(losses, unit="monetary").calculate(0.95)
    monte_carlo = (
        MonteCarloRiskModel(n_simulations=100_000, random_seed=2026)
        .fit(losses, unit="monetary")
        .calculate(0.95)
    )
    garch = (
        GARCHRiskModel()
        .fit(
            portfolio_result.portfolio_returns,
            unit="monetary",
            portfolio_value=float(portfolio_result.portfolio_value.iloc[-1]),
        )
        .calculate(0.95)
    )

    results = [historical, gaussian, student_t, monte_carlo, garch]
    assert [result.sample_size for result in results] == [
        4184,
        4184,
        4184,
        100_000,
        4184,
    ]
    assert all(result.confidence_level == 0.95 for result in results)
    assert all(result.unit is RiskUnit.MONETARY for result in results)
    assert all(
        np.isfinite([result.var, result.expected_shortfall]).all() for result in results
    )
    assert all(result.expected_shortfall >= result.var for result in results)


def test_arch_failure_is_reported(monkeypatch) -> None:
    def fail_fit(*args, **kwargs):
        raise ValueError("deliberate optimizer failure")

    class FakeModel:
        fit = fail_fit

    monkeypatch.setattr(
        "riskchallenger.risk.garch.arch_model", lambda *args, **kwargs: FakeModel()
    )
    with pytest.raises(RuntimeError, match="fitting failed"):
        GARCHRiskModel().fit(make_garch_returns(), unit="relative")
