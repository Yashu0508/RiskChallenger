from __future__ import annotations

import pandas as pd
import pytest
from dashboard.data_access import (
    DEFAULT_ASSETS,
    DEFAULT_WEIGHTS,
    DEMO_VALUE,
    build_stress_results,
    create_portfolio,
    discover_asset_universe,
    format_risk_results,
    format_stress_results,
    is_reference_portfolio,
    load_clean_prices,
    load_phase4_snapshot,
    load_phase7_snapshot,
    validate_dashboard_weights,
)

from riskchallenger.portfolio.engine import load_return_dataset
from riskchallenger.risk import RiskResult, RiskUnit
from riskchallenger.stress import StressTestEngine, UniformShock


def test_dashboard_loads_validated_phase_snapshots() -> None:
    phase4 = load_phase4_snapshot()
    phase7 = load_phase7_snapshot()
    assert phase4["available"] is True
    assert phase4["period"]["forecast_count"] == 3684
    assert phase4["champion_challenger_summary"]["champion"] == "garch_1_1_gaussian"
    assert phase7["available"] is True
    assert phase7["forecast_count"] == 180
    assert (
        "recent-period diagnostic"
        in phase7["champion_challenger_diagnostic"]["note"].lower()
    )


def test_missing_dashboard_artifact_is_reported_as_unavailable(tmp_path) -> None:
    missing = tmp_path / "not-created.json"
    phase4 = load_phase4_snapshot(missing)
    phase7 = load_phase7_snapshot(missing)
    assert phase4["available"] is False
    assert phase7["available"] is False
    assert "not found" in phase4["message"]


def test_portfolio_selection_uses_phase2_weight_validation_without_normalization() -> (
    None
):
    portfolio = create_portfolio(DEFAULT_ASSETS, DEFAULT_WEIGHTS, DEMO_VALUE)
    assert portfolio.weight_vector == (0.5, 0.3, 0.2)
    with pytest.raises(ValueError, match="sum to 1.0"):
        create_portfolio(
            DEFAULT_ASSETS,
            {"HDFCBANK.NS": 0.4, "ICICIBANK.NS": 0.3, "RELIANCE.NS": 0.2},
            DEMO_VALUE,
        )
    with pytest.raises(ValueError, match="missing weights"):
        create_portfolio(DEFAULT_ASSETS, {"HDFCBANK.NS": 1.0}, DEMO_VALUE)
    assert is_reference_portfolio(portfolio)
    alternative = create_portfolio(
        ["TCS.NS", "INFY.NS"], {"TCS.NS": 0.6, "INFY.NS": 0.4}, DEMO_VALUE
    )
    assert not is_reference_portfolio(alternative)


def test_dashboard_discovers_complete_canonical_universe() -> None:
    assets, coverage = discover_asset_universe()
    returns = load_return_dataset()
    prices = load_clean_prices()
    assert len(assets) == len(set(returns.columns) & set(prices.columns)) == 50
    assert set(assets) == set(returns.columns)
    assert coverage["Available"].all()
    assert DEFAULT_ASSETS[0] in assets


def test_dashboard_weight_validation_is_explicit_and_never_normalizes() -> None:
    assert validate_dashboard_weights(DEFAULT_ASSETS, DEFAULT_WEIGHTS)[0]
    valid, message = validate_dashboard_weights(
        ["TCS.NS", "INFY.NS"], {"TCS.NS": 0.6, "INFY.NS": 0.3}
    )
    assert not valid
    assert "sum to 100%" in message
    valid, message = validate_dashboard_weights(["TCS.NS"], {"TCS.NS": float("nan")})
    assert not valid
    assert "finite" in message


def test_risk_result_formatting_preserves_model_value_unit_and_sample() -> None:
    item = RiskResult(
        model_name="historical",
        confidence_level=0.95,
        var=25_000.0,
        expected_shortfall=30_000.0,
        sample_size=3684,
        loss_convention="positive values are losses",
        unit=RiskUnit.MONETARY,
    )
    frame = format_risk_results([item])
    assert frame.iloc[0].to_dict() == {
        "Model": "historical",
        "Confidence": 0.95,
        "VaR": 25_000.0,
        "ES": 30_000.0,
        "Unit": "monetary",
        "Sample size": 3684,
    }


def test_stress_result_formatting_and_demo_event_scenarios() -> None:
    portfolio = create_portfolio(DEFAULT_ASSETS, DEFAULT_WEIGHTS, DEMO_VALUE)
    result = StressTestEngine().run_scenario(
        portfolio, UniformShock(-0.05, name="Uniform -5%")
    )
    frame = format_stress_results([result])
    assert frame.loc[0, "pnl"] == pytest.approx(-50_000)
    assert frame.loc[0, "loss_percentage"] == pytest.approx(5)
    scenarios = build_stress_results(portfolio)
    assert [item.scenario_name for item in scenarios] == [
        "Uniform -5%",
        "Uniform -10%",
        "Uniform -20%",
        "Asset-specific adverse",
        "Global Financial Crisis replay",
        "COVID-19 crash replay",
    ]
    attribution = scenarios[3].asset_impacts_frame
    assert attribution["pnl_contribution"].sum() == pytest.approx(scenarios[3].pnl)
    assert isinstance(scenarios[-1].start_date, pd.Timestamp)
