"""Synthetic tests for scenario-based portfolio stress testing."""

import numpy as np
import pandas as pd
import pytest

from riskchallenger.portfolio import Portfolio
from riskchallenger.stress import (
    AssetSpecificShock,
    HistoricalReplayScenario,
    JointShock,
    StressTestEngine,
    UniformShock,
    VolatilityShock,
)


def one_asset_portfolio(value: float = 100.0) -> Portfolio:
    return Portfolio(["A"], {"A": 1.0}, value)


def test_uniform_shock_has_exact_value_pnl_and_loss() -> None:
    result = StressTestEngine().run_scenario(one_asset_portfolio(), UniformShock(-0.10))
    assert result.base_value == pytest.approx(100)
    assert result.stressed_value == pytest.approx(90)
    assert result.pnl == pytest.approx(-10)
    assert result.loss == pytest.approx(10)
    assert result.loss_percentage == pytest.approx(10)
    assert result.asset_impacts[0].pnl_contribution == pytest.approx(-10)


def test_asset_specific_shock_weighted_contributions_reconcile() -> None:
    portfolio = Portfolio(
        ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"],
        {"HDFCBANK.NS": 0.50, "ICICIBANK.NS": 0.30, "RELIANCE.NS": 0.20},
        1_000_000,
    )
    scenario = AssetSpecificShock(
        {
            "HDFCBANK.NS": -0.10,
            "ICICIBANK.NS": -0.15,
            "RELIANCE.NS": -0.20,
        }
    )
    result = StressTestEngine().run_scenario(portfolio, scenario)
    assert [item.pnl_contribution for item in result.asset_impacts] == pytest.approx(
        [-50_000, -45_000, -40_000]
    )
    assert result.pnl == pytest.approx(-135_000)
    assert result.stressed_value == pytest.approx(865_000)
    assert sum(item.pnl_contribution for item in result.asset_impacts) == pytest.approx(
        result.pnl
    )


def test_historical_replay_uses_cumulative_endpoint_price_returns() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    scenario = HistoricalReplayScenario.from_price_window(
        prices, "known_window", "2020-01-02", "2020-01-03", ["A"]
    )
    result = StressTestEngine().run_scenario(one_asset_portfolio(1_000), scenario)
    assert dict(scenario.asset_returns)["A"] == pytest.approx(-0.01)
    assert result.pnl == pytest.approx(-10)
    assert result.start_date == pd.Timestamp("2020-01-02")
    assert result.end_date == pd.Timestamp("2020-01-03")
    assert "Today's portfolio weights" in " ".join(result.assumptions)


def test_historical_return_vector_replays_current_weights() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 0.6, "B": 0.4}, 10_000)
    scenario = HistoricalReplayScenario(
        {"A": -0.08, "B": -0.12}, name="historical_vector"
    )
    result = StressTestEngine().run_scenario(portfolio, scenario)
    assert result.pnl == pytest.approx(-960)
    assert result.stressed_value == pytest.approx(9_040)


def test_volatility_shock_scales_an_explicit_reference_realization() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 0.5, "B": 0.5}, 1_000)
    scenario = VolatilityShock({"A": -0.10, "B": 0.02}, volatility_multiplier=2.0)
    result = StressTestEngine().run_scenario(portfolio, scenario)
    assert [item.return_shock for item in result.asset_impacts] == pytest.approx(
        [-0.20, 0.04]
    )
    assert result.pnl == pytest.approx(-80)


def test_joint_shock_is_common_plus_idiosyncratic() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 0.5, "B": 0.5}, 1_000)
    result = StressTestEngine().run_scenario(
        portfolio,
        JointShock(-0.10, {"A": -0.02, "B": 0.01}),
    )
    assert [item.return_shock for item in result.asset_impacts] == pytest.approx(
        [-0.12, -0.09]
    )
    assert result.pnl == pytest.approx(-105)


def test_unknown_and_duplicate_assets_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown portfolio assets"):
        StressTestEngine().run_scenario(
            one_asset_portfolio(), AssetSpecificShock({"UNKNOWN": -0.1})
        )
    with pytest.raises(ValueError, match="duplicate asset"):
        AssetSpecificShock([("A", -0.1), ("A", -0.2)])
    with pytest.raises(ValueError, match="duplicate"):
        Portfolio(["A", "A"], {"A": 1.0}, 100)


@pytest.mark.parametrize("shock", [float("nan"), float("inf"), -1.0, -1.2])
def test_invalid_or_price_destroying_shocks_are_rejected(shock: float) -> None:
    with pytest.raises(ValueError):
        UniformShock(shock)


def test_asset_scenario_requires_complete_asset_coverage() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 0.5, "B": 0.5}, 100)
    with pytest.raises(ValueError, match="missing shocks"):
        StressTestEngine().run_scenario(portfolio, AssetSpecificShock({"A": -0.1}))


def test_zero_shock_leaves_portfolio_unchanged() -> None:
    result = StressTestEngine().run_scenario(
        one_asset_portfolio(1234.5), UniformShock(0.0)
    )
    assert result.stressed_value == pytest.approx(result.base_value)
    assert result.pnl == pytest.approx(0.0)
    assert result.loss == pytest.approx(0.0)


def test_severe_shock_cannot_create_zero_or_negative_asset_price() -> None:
    with pytest.raises(ValueError, match="greater than -1"):
        AssetSpecificShock({"A": -1.0})
    with pytest.raises(ValueError, match="volatility-scaled return"):
        VolatilityShock({"A": -0.6}, volatility_multiplier=2.0).resolve(["A"])


def test_invalid_volatility_multiplier_and_joint_shock_are_rejected() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        VolatilityShock({"A": -0.1}, 0.5)
    with pytest.raises(ValueError, match="combined shock"):
        JointShock(-0.9, {"A": -0.2}).resolve(["A"])
    with pytest.raises(ValueError, match="unknown assets"):
        JointShock(-0.1, {"UNKNOWN": -0.2}).resolve(["A"])


def test_invalid_dates_and_missing_historical_endpoints_are_rejected() -> None:
    with pytest.raises(ValueError, match="both start_date and end_date"):
        HistoricalReplayScenario({"A": -0.1}, start_date="2020-01-01")
    with pytest.raises(ValueError, match="must precede"):
        HistoricalReplayScenario(
            {"A": -0.1}, start_date="2020-02-01", end_date="2020-01-01"
        )
    prices = pd.DataFrame(
        {"A": [100.0, 110.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    with pytest.raises(ValueError, match="no observation inside"):
        HistoricalReplayScenario.from_price_window(
            prices, "no_window", "2020-01-03", "2020-01-04", ["A"]
        )


def test_invalid_base_value_and_nonpositive_portfolio_outcome_fail() -> None:
    with pytest.raises(ValueError, match="base_value"):
        StressTestEngine().run_scenario(one_asset_portfolio(), UniformShock(0), 0)
    leveraged = Portfolio(["A", "B"], {"A": 2.0, "B": -1.0}, 100, long_only=False)
    with pytest.raises(ValueError, match="non-positive portfolio value"):
        StressTestEngine().run_scenario(
            leveraged, AssetSpecificShock({"A": -0.99, "B": 0.1})
        )


def test_all_impacts_reconcile_to_total_pnl() -> None:
    portfolio = Portfolio(["A", "B", "C"], {"A": 0.2, "B": 0.3, "C": 0.5}, 987_654)
    result = StressTestEngine().run_scenario(
        portfolio, AssetSpecificShock({"A": 0.04, "B": -0.11, "C": -0.18})
    )
    assert np.isclose(
        sum(item.pnl_contribution for item in result.asset_impacts),
        result.pnl,
        atol=1e-8,
    )
    assert result.loss == pytest.approx(-result.pnl)


def test_actual_canonical_data_supports_both_crisis_windows() -> None:
    from pathlib import Path

    prices = pd.read_csv(
        Path("data/processed/nse50_prices_clean.csv"), parse_dates=["date"]
    ).set_index("date")
    for name, start, end in (
        ("global_financial_crisis", "2008-09-15", "2009-03-09"),
        ("covid_crash", "2020-02-19", "2020-03-23"),
    ):
        scenario = HistoricalReplayScenario.from_price_window(
            prices,
            name,
            start,
            end,
            ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"],
        )
        assert len(scenario.asset_returns) == 3
        assert all(
            np.isfinite(value) and value > -1 for _, value in scenario.asset_returns
        )
