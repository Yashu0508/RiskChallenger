# Local REST API

RiskChallenger exposes a local FastAPI interface over the established Phase 2–6
engines and saved Phase 4/5 and Phase 7 summaries. It is a transport layer: it
does not implement portfolio or risk formulas, rewrite datasets, or rerun the
expensive expanding-window backtest.

## Run

Install the project and start Uvicorn from the repository root:

```bash
python -m pip install -e ".[dev]"
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

The service binds to loopback by default in this documented command. The
interactive OpenAPI interface is at `http://127.0.0.1:8000/docs`; health is
available at `/health`. No external service, API key, or network connection is
used by the API itself.

## Endpoints

- `GET /health` — lightweight process/version health check.
- `POST /portfolio/analyze` — assets, explicit weights, and starting value;
  returns Phase 2 portfolio statistics and sample dates.
- `POST /risk/var-es` — select `historical`, `gaussian`, `student_t`,
  `monte_carlo`, or `garch`, and provide confidence. Outputs retain the
  established positive-loss convention and monetary units.
- `POST /risk/compare` — calculate all five one-period models in one request;
  it does not invoke rolling refits or backtesting.
- `POST /stress/run` — run uniform -5/-10/-20%, asset-specific, GFC replay, or
  COVID replay using Phase 6 scenario definitions and attribution.
- `GET /models/champion` — reads the persisted full-period Phase 4/5 snapshot.
- `GET /models/ml-diagnostic` — reads the separate recent-period Phase 7
  diagnostic snapshot.

Portfolio payloads require an asset list, a matching asset-to-weight mapping,
and a positive initial value. Weights are never normalized; Phase 2 validates
them and unknown assets. Risk inputs choose one model at a time. The GARCH risk
request is monetary and uses the explicitly provided `initial_value` as the
portfolio value. An asset-specific scenario can supply `asset_shocks`; without
it, the demonstration shocks apply only to the three demonstration assets.

For example:

```json
{
  "assets": ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"],
  "weights": {"HDFCBANK.NS": 0.5, "ICICIBANK.NS": 0.3, "RELIANCE.NS": 0.2},
  "initial_value": 1000000,
  "model": "historical",
  "confidence_level": 0.95
}
```

The request above is sent to `POST /risk/var-es`. Invalid requests return HTTP
422; unavailable local project files return HTTP 503. Saved summary routes
return `available: false` when an artifact has not been persisted. The API
does not expose filesystem paths or stack traces in error responses. Structural
request errors include field-level validation details; internal data validation
failures receive a generic response.

## Data and scope

Inputs are the repository's canonical processed returns/prices, consumed
read-only. `/models/*` reads compact checked-in report artifacts and does not
refit models. The API does not expose full backtest execution, ML fitting,
external integrations, or later-phase functionality.
