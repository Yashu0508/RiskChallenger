"""Thin orchestration over the established RiskChallenger engines."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from riskchallenger.data.loader import repository_root
from riskchallenger.portfolio.engine import calculate_portfolio, load_return_dataset
from riskchallenger.portfolio.portfolio import Portfolio
from riskchallenger.risk import (
    GARCHRiskModel,
    GaussianRiskModel,
    HistoricalRiskModel,
    MonteCarloRiskModel,
    RiskUnit,
    StudentTRiskModel,
)
from riskchallenger.stress import (
    AssetSpecificShock,
    HistoricalReplayScenario,
    StressTestEngine,
    UniformShock,
)

PRICE_PATH = Path("data/processed/nse50_prices_clean.csv")
REPORTS = Path("data/reports")
DEFAULT_ASSET_SHOCKS = {
    "HDFCBANK.NS": -0.10,
    "ICICIBANK.NS": -0.15,
    "RELIANCE.NS": -0.20,
}


def get_returns() -> pd.DataFrame:
    """Load validated Phase 1 returns from the repository-relative location."""
    return load_return_dataset()


def get_prices() -> pd.DataFrame:
    """Load immutable canonical prices for historical replay scenarios."""
    path = repository_root() / PRICE_PATH
    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame:
        raise ValueError("canonical price dataset is unavailable or malformed")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame.set_index("date")


def portfolio_analysis(request: Any, returns: pd.DataFrame) -> dict[str, Any]:
    portfolio = Portfolio(request.assets, request.weights, request.initial_value)
    result = calculate_portfolio(portfolio, returns)
    return {
        "assets": list(portfolio.assets),
        "weights": dict(portfolio.weights),
        "initial_value": portfolio.initial_value,
        "final_value": float(result.portfolio_value.iloc[-1]),
        "observations": len(result.portfolio_returns),
        "first_date": result.portfolio_returns.index[0].date().isoformat(),
        "last_date": result.portfolio_returns.index[-1].date().isoformat(),
        "statistics": result.statistics.to_dict(),
    }


def risk_calculation(request: Any, returns: pd.DataFrame) -> dict[str, Any]:
    portfolio = Portfolio(request.assets, request.weights, request.initial_value)
    result = calculate_portfolio(portfolio, returns)
    models = {
        "historical": HistoricalRiskModel,
        "gaussian": GaussianRiskModel,
        "student_t": StudentTRiskModel,
    }
    if request.model == "monte_carlo":
        model = MonteCarloRiskModel(request.n_simulations, request.random_seed).fit(
            result.loss.to_numpy(), unit=RiskUnit.MONETARY
        )
    elif request.model == "garch":
        model = GARCHRiskModel().fit(
            result.portfolio_returns.to_numpy(),
            unit=RiskUnit.MONETARY,
            portfolio_value=request.initial_value,
        )
    else:
        model = models[request.model]().fit(
            result.loss.to_numpy(), unit=RiskUnit.MONETARY
        )
    risk = model.calculate(request.confidence_level)
    return {
        "model": risk.model_name,
        "confidence_level": risk.confidence_level,
        "var": risk.var,
        "expected_shortfall": risk.expected_shortfall,
        "sample_size": risk.sample_size,
        "unit": risk.unit.value,
        "loss_convention": risk.loss_convention,
        "model_parameters": risk.model_parameters,
    }


def stress_calculation(request: Any, prices: pd.DataFrame) -> dict[str, Any]:
    portfolio = Portfolio(request.assets, request.weights, request.initial_value)
    if request.scenario.startswith("uniform_"):
        pct = int(request.scenario.split("_")[1])
        scenario = UniformShock(-pct / 100, name=f"Uniform -{pct}%")
    elif request.scenario == "asset_specific_adverse":
        shocks = request.asset_shocks or {
            asset: DEFAULT_ASSET_SHOCKS[asset]
            for asset in request.assets
            if asset in DEFAULT_ASSET_SHOCKS
        }
        scenario = AssetSpecificShock(shocks, name="Asset-specific adverse")
    else:
        start, end, name = (
            ("2008-09-15", "2009-03-09", "Global Financial Crisis replay")
            if request.scenario == "gfc_replay"
            else ("2020-02-19", "2020-03-23", "COVID-19 crash replay")
        )
        scenario = HistoricalReplayScenario.from_price_window(
            prices,
            name=name,
            start_date=start,
            end_date=end,
            assets=portfolio.assets,
        )
    output = StressTestEngine().run_scenario(
        portfolio, scenario, base_value=request.base_value
    )
    return {
        "scenario_name": output.scenario_name,
        "scenario_type": output.scenario_type,
        "base_value": output.base_value,
        "stressed_value": output.stressed_value,
        "pnl": output.pnl,
        "loss": output.loss,
        "loss_percentage": output.loss_percentage,
        "asset_impacts": [impact.__dict__ for impact in output.asset_impacts],
        "assumptions": list(output.assumptions),
        "start_date": output.start_date,
        "end_date": output.end_date,
    }


def load_summary(filename: str) -> dict[str, Any]:
    """Read existing validated Phase 4/5 or Phase 7 snapshots without refitting."""
    path = repository_root() / REPORTS / filename
    if not path.is_file():
        return {"available": False, "message": "validated summary artifact unavailable"}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("validated summary artifact is malformed") from error
    if not isinstance(content, dict):
        raise ValueError("validated summary artifact must be a JSON object")
    content["available"] = True
    return content
