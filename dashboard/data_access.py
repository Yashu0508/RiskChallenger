"""Dashboard data access and presentation transforms.

All portfolio, risk, and stress calculations delegate to the project's existing
Phase 2, 3, and 6 APIs. Persisted backtest snapshots avoid refitting Phase 4.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskchallenger.data.loader import repository_root
from riskchallenger.portfolio.engine import (
    PortfolioCalculationResult,
    calculate_portfolio,
    load_return_dataset,
)
from riskchallenger.portfolio.portfolio import Portfolio
from riskchallenger.risk import (
    GARCHRiskModel,
    GaussianRiskModel,
    HistoricalRiskModel,
    MonteCarloRiskModel,
    RiskResult,
    RiskUnit,
    StudentTRiskModel,
)
from riskchallenger.stress import (
    AssetSpecificShock,
    HistoricalReplayScenario,
    StressResult,
    StressTestEngine,
    UniformShock,
)

DEFAULT_ASSETS = ("HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS")
DEFAULT_WEIGHTS = {
    "HDFCBANK.NS": 0.50,
    "ICICIBANK.NS": 0.30,
    "RELIANCE.NS": 0.20,
}
DEMO_VALUE = 1_000_000.0
REPORT_DIRECTORY = Path("data/reports")
PHASE4_SNAPSHOT = REPORT_DIRECTORY / "dashboard_phase4_summary.json"
PHASE7_SNAPSHOT = REPORT_DIRECTORY / "dashboard_phase7_diagnostic.json"
PRICE_DATASET = Path("data/processed/nse50_prices_clean.csv")


def load_snapshot(path: Path | str) -> dict[str, Any]:
    """Load a checked-in JSON result snapshot with a clear missing-file error."""
    source = Path(path)
    if not source.is_absolute():
        source = repository_root() / source
    if not source.is_file():
        raise FileNotFoundError(f"Dashboard result snapshot not found: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Dashboard result snapshot is invalid JSON: {source}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError(f"Dashboard result snapshot must be a JSON object: {source}")
    return data


def load_phase4_snapshot(path: Path | str = PHASE4_SNAPSHOT) -> dict[str, Any]:
    """Load previously validated full-period Phase 4 summary results."""
    try:
        snapshot = load_snapshot(path)
    except FileNotFoundError as error:
        return {"available": False, "message": str(error)}
    if not snapshot.get("models") or not snapshot.get("period"):
        raise ValueError("Phase 4 snapshot requires period and model summaries")
    snapshot["available"] = True
    return snapshot


def load_phase7_snapshot(path: Path | str = PHASE7_SNAPSHOT) -> dict[str, Any]:
    """Load the explicitly labeled recent-period Phase 7 diagnostic."""
    try:
        snapshot = load_snapshot(path)
    except FileNotFoundError as error:
        return {"available": False, "message": str(error)}
    if not snapshot.get("models") or not snapshot.get("source"):
        raise ValueError("Phase 7 snapshot requires source and model results")
    snapshot["available"] = True
    return snapshot


def load_clean_prices(path: Path | str = PRICE_DATASET) -> pd.DataFrame:
    """Read the immutable canonical price file for historical stress replay."""
    source = Path(path)
    if not source.is_absolute():
        source = repository_root() / source
    if not source.is_file():
        raise FileNotFoundError(f"Canonical price dataset not found: {source}")
    frame = pd.read_csv(source)
    if frame.empty or "date" not in frame.columns:
        raise ValueError("Canonical prices must have observations and a date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.set_index("date")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Canonical price dates must be unique and chronological")
    return frame


def discover_asset_universe(
    *,
    returns: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    minimum_observations: int = 2,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Discover assets present in canonical returns and prices with usable data.

    Availability is derived from dataset columns, not a dashboard-maintained
    ticker list. Coverage reports finite return/price counts so the UI can
    explain why an asset may not support a calculation.
    """
    if minimum_observations < 1:
        raise ValueError("minimum_observations must be positive")
    return_frame = returns if returns is not None else load_return_dataset()
    price_frame = prices if prices is not None else load_clean_prices()
    common = [asset for asset in return_frame.columns if asset in price_frame.columns]
    coverage: list[dict[str, Any]] = []
    available: list[str] = []
    for asset in common:
        asset_returns = pd.to_numeric(return_frame[asset], errors="coerce").to_numpy(
            dtype=float
        )
        asset_prices = pd.to_numeric(price_frame[asset], errors="coerce").to_numpy(
            dtype=float
        )
        valid_returns = np.isfinite(asset_returns)
        valid_prices = np.isfinite(asset_prices) & (asset_prices > 0)
        return_count = int(valid_returns.sum())
        price_count = int(valid_prices.sum())
        is_available = min(return_count, price_count) >= minimum_observations
        coverage.append(
            {
                "Asset": asset,
                "Return observations": return_count,
                "Missing/invalid returns": int(len(asset_returns) - return_count),
                "Price observations": price_count,
                "Missing/invalid prices": int(len(asset_prices) - price_count),
                "Available": is_available,
            }
        )
        if is_available:
            available.append(asset)
    if not coverage:
        raise ValueError("No assets are shared by canonical prices and returns")
    return tuple(available), pd.DataFrame(coverage)


def validate_dashboard_weights(
    assets: tuple[str, ...] | list[str],
    weights: dict[str, float],
    *,
    tolerance: float = 1e-8,
) -> tuple[bool, str]:
    """Check finite, matching, long-only weights without normalizing them."""
    if not assets:
        return False, "Select at least one asset."
    if len(assets) != len(set(assets)):
        return False, "Duplicate assets are not allowed."
    if set(assets) != set(weights):
        return False, "Provide exactly one weight for each selected asset."
    values = np.asarray([weights[asset] for asset in assets], dtype=float)
    if not np.isfinite(values).all():
        return False, "Weights must be finite numeric values."
    if (values < 0).any():
        return False, "Weights must be non-negative for a long-only portfolio."
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=tolerance):
        return False, f"Weights total {total:.6f}; adjust them to sum to 100%."
    return True, "Weights validated; no normalization applied."


def create_portfolio(
    assets: tuple[str, ...] | list[str],
    weights: dict[str, float],
    initial_value: float,
) -> Portfolio:
    """Build the existing validated Phase 2 portfolio; never normalize weights."""
    return Portfolio(assets=assets, weights=weights, initial_value=initial_value)


def is_reference_portfolio(portfolio: Portfolio) -> bool:
    """Whether the portfolio exactly matches the saved Phase 4/5 reference run."""
    return (
        set(portfolio.assets) == set(DEFAULT_ASSETS)
        and all(
            np.isclose(
                portfolio.weights[asset],
                DEFAULT_WEIGHTS[asset],
                rtol=0.0,
                atol=1e-10,
            )
            for asset in DEFAULT_ASSETS
        )
        and np.isclose(portfolio.initial_value, DEMO_VALUE, rtol=0.0, atol=1e-8)
    )


def build_portfolio_result(portfolio: Portfolio) -> PortfolioCalculationResult:
    """Calculate portfolio time series using existing returns and Phase 2 API."""
    returns = load_return_dataset()
    return calculate_portfolio(portfolio, returns)


def calculate_risk_results(
    result: PortfolioCalculationResult,
    confidence_level: float,
    *,
    monte_carlo_simulations: int = 10_000,
    random_seed: int = 2026,
) -> list[RiskResult]:
    """Delegate VaR/ES calculations to the Phase 3 risk-model implementations."""
    losses = result.loss.to_numpy(dtype=float)
    output = [
        HistoricalRiskModel()
        .fit(losses, unit=RiskUnit.MONETARY)
        .calculate(confidence_level),
        GaussianRiskModel()
        .fit(losses, unit=RiskUnit.MONETARY)
        .calculate(confidence_level),
        StudentTRiskModel()
        .fit(losses, unit=RiskUnit.MONETARY)
        .calculate(confidence_level),
        MonteCarloRiskModel(
            n_simulations=monte_carlo_simulations, random_seed=random_seed
        )
        .fit(losses, unit=RiskUnit.MONETARY)
        .calculate(confidence_level),
    ]
    garch = GARCHRiskModel().fit(
        result.portfolio_returns.to_numpy(dtype=float),
        unit=RiskUnit.MONETARY,
        portfolio_value=float(result.portfolio_value.iloc[-1]),
    )
    output.append(garch.calculate(confidence_level))
    return output


def format_risk_results(results: list[RiskResult]) -> pd.DataFrame:
    """Transform typed Phase 3 risk results into a dashboard-ready table."""
    return pd.DataFrame(
        [
            {
                "Model": item.model_name,
                "Confidence": item.confidence_level,
                "VaR": item.var,
                "ES": item.expected_shortfall,
                "Unit": item.unit.value,
                "Sample size": item.sample_size,
            }
            for item in results
        ]
    )


def build_stress_results(
    portfolio: Portfolio,
    *,
    prices: pd.DataFrame | None = None,
    base_value: float | None = None,
    asset_shocks: dict[str, float] | None = None,
) -> list[StressResult]:
    """Run named deterministic scenarios using the Phase 6 stress engine."""
    asset_shock_values = {
        "HDFCBANK.NS": -0.10,
        "ICICIBANK.NS": -0.15,
        "RELIANCE.NS": -0.20,
    }
    if asset_shocks is None:
        if any(asset not in asset_shock_values for asset in portfolio.assets):
            raise ValueError(
                "Provide an explicit asset shock for every selected asset."
            )
        resolved_asset_shocks = {
            asset: asset_shock_values[asset] for asset in portfolio.assets
        }
    else:
        resolved_asset_shocks = asset_shocks
    scenarios = [
        UniformShock(-0.05, name="Uniform -5%"),
        UniformShock(-0.10, name="Uniform -10%"),
        UniformShock(-0.20, name="Uniform -20%"),
        AssetSpecificShock(
            resolved_asset_shocks,
            name="Asset-specific adverse",
        ),
    ]
    canonical_prices = prices if prices is not None else load_clean_prices()
    historical_windows = (
        ("Global Financial Crisis replay", "2008-09-15", "2009-03-09"),
        ("COVID-19 crash replay", "2020-02-19", "2020-03-23"),
    )
    for name, start, end in historical_windows:
        scenarios.append(
            HistoricalReplayScenario.from_price_window(
                canonical_prices,
                name=name,
                start_date=start,
                end_date=end,
                assets=portfolio.assets,
            )
        )
    engine = StressTestEngine()
    return [
        engine.run_scenario(portfolio, scenario, base_value=base_value)
        for scenario in scenarios
    ]


def format_stress_results(results: list[StressResult]) -> pd.DataFrame:
    """Format typed Phase 6 stress results without redoing scenario math."""
    return pd.DataFrame([item.summary() for item in results])


def overview_metrics(result: PortfolioCalculationResult) -> dict[str, float]:
    """Combine existing Phase 2 statistics with standard annualized summaries."""
    returns = result.portfolio_returns
    path = np.concatenate(
        ([result.statistics.initial_portfolio_value], result.portfolio_value.to_numpy())
    )
    running_peak = np.maximum.accumulate(path)
    drawdown = path / running_peak - 1.0
    return {
        **result.statistics.to_dict(),
        "annualized_mean_return": float(returns.mean() * 252.0),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252.0)),
        "maximum_drawdown": float(drawdown.min()),
    }
