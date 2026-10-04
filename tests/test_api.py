"""HTTP-level tests for the local FastAPI boundary."""

import numpy as np
import pandas as pd
import pytest
from api.app import app
from api.services import get_prices, get_returns
from fastapi.testclient import TestClient

ASSETS = ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"]
WEIGHTS = {ASSETS[0]: 0.5, ASSETS[1]: 0.3, ASSETS[2]: 0.2}


@pytest.fixture
def client() -> TestClient:
    dates = pd.date_range("2020-01-01", periods=180, freq="B")
    t = np.arange(len(dates), dtype=float)
    returns = pd.DataFrame(
        {
            "date": dates,
            ASSETS[0]: 0.001 * np.sin(t / 5) + 0.0002,
            ASSETS[1]: 0.0015 * np.cos(t / 7),
            ASSETS[2]: 0.001 * np.sin(t / 9) - 0.0001,
        }
    ).set_index("date")
    returns.iloc[0] = np.nan
    price_dates = pd.date_range("2008-01-01", "2020-04-01", freq="B")
    prices = pd.DataFrame(
        {
            asset: 100.0
            + np.arange(len(price_dates)) * 0.01
            + np.sin(np.arange(len(price_dates)) / (index + 3))
            for index, asset in enumerate(ASSETS)
        },
        index=price_dates,
    )
    app.dependency_overrides[get_returns] = lambda: returns
    app.dependency_overrides[get_prices] = lambda: prices
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def portfolio_body() -> dict:
    return {
        "assets": ASSETS.copy(),
        "weights": WEIGHTS.copy(),
        "initial_value": 1_000_000,
    }


def test_health_and_openapi(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/openapi.json").status_code == 200


def test_portfolio_analysis_and_weight_validation(client: TestClient) -> None:
    response = client.post("/portfolio/analyze", json=portfolio_body())
    assert response.status_code == 200
    assert response.json()["observations"] == 179
    invalid = portfolio_body()
    invalid["weights"][ASSETS[0]] = 0.4
    assert client.post("/portfolio/analyze", json=invalid).status_code == 422


def test_duplicate_unknown_and_nonfinite_inputs(client: TestClient) -> None:
    body = portfolio_body()
    body["assets"] = [ASSETS[0], ASSETS[0]]
    assert client.post("/portfolio/analyze", json=body).status_code == 422
    body = portfolio_body()
    body["assets"] = ["NOT.ASSET"]
    body["weights"] = {"NOT.ASSET": 1}
    assert client.post("/portfolio/analyze", json=body).status_code == 422
    body = portfolio_body()
    body["initial_value"] = "nan"
    assert client.post("/portfolio/analyze", json=body).status_code == 422


@pytest.mark.parametrize(
    "model", ["historical", "gaussian", "student_t", "monte_carlo", "garch"]
)
def test_risk_endpoint_uses_existing_models(client: TestClient, model: str) -> None:
    body = portfolio_body() | {"model": model, "confidence_level": 0.95}
    response = client.post("/risk/var-es", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["unit"] == "monetary"
    assert result["sample_size"] > 0
    assert np.isfinite([result["var"], result["expected_shortfall"]]).all()


def test_risk_invalid_confidence_and_model(client: TestClient) -> None:
    body = portfolio_body() | {"model": "gaussian", "confidence_level": 1.0}
    assert client.post("/risk/var-es", json=body).status_code == 422
    body = portfolio_body() | {"model": "unknown", "confidence_level": 0.95}
    assert client.post("/risk/var-es", json=body).status_code == 422


def test_risk_comparison_endpoint(client: TestClient) -> None:
    body = portfolio_body() | {
        "model": "historical",
        "confidence_level": 0.95,
        "n_simulations": 250,
    }
    response = client.post("/risk/compare", json=body)
    assert response.status_code == 200, response.text
    assert len(response.json()) == 5
    assert {item["model"] for item in response.json()} == {
        "historical",
        "gaussian",
        "student_t",
        "monte_carlo",
        "garch_1_1_gaussian",
    }


@pytest.mark.parametrize(
    "scenario", ["uniform_5", "uniform_10", "uniform_20", "asset_specific_adverse"]
)
def test_deterministic_stress_endpoint(client: TestClient, scenario: str) -> None:
    body = portfolio_body() | {"scenario": scenario}
    response = client.post("/stress/run", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["asset_impacts"]) == 3
    assert result["stressed_value"] == pytest.approx(
        result["base_value"] + result["pnl"]
    )
    assert sum(
        item["pnl_contribution"] for item in result["asset_impacts"]
    ) == pytest.approx(result["pnl"])


def test_invalid_stress_scenario(client: TestClient) -> None:
    body = portfolio_body() | {"scenario": "invented"}
    assert client.post("/stress/run", json=body).status_code == 422


@pytest.mark.parametrize("scenario", ["gfc_replay", "covid_replay"])
def test_historical_replay_endpoint(client: TestClient, scenario: str) -> None:
    body = portfolio_body() | {"scenario": scenario}
    response = client.post("/stress/run", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["start_date"] is not None


def test_persisted_phase_summaries(client: TestClient) -> None:
    champion = client.get("/models/champion").json()
    diagnostic = client.get("/models/ml-diagnostic").json()
    assert champion["available"] is True
    assert diagnostic["available"] is True
    assert champion["champion_challenger_summary"]["champion"] == "garch_1_1_gaussian"
    assert "recent" in diagnostic["champion_challenger_diagnostic"]["note"].lower()
