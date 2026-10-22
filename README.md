# RiskChallenger

**AI-Assisted Quantitative Risk Model Validation & Challenger Framework**

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://riskchallenger-ltg7attq856fpddede4ue2.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Testing](https://img.shields.io/badge/Testing-pytest-0a9edc)](https://pytest.org/)

RiskChallenger is a local quantitative risk-management and model-validation platform for NSE equity portfolios. It connects historical market data, portfolio construction, loss-based risk models, out-of-sample backtesting, scenario stress testing, and model diagnostics in one reproducible workflow.

**Live application:** [Open the RiskChallenger dashboard](https://riskchallenger-ltg7attq856fpddede4ue2.streamlit.app/)

## What the system does

RiskChallenger is designed around a practical risk-management question:

> How should a portfolio’s risk estimates be calculated, validated, challenged, and communicated?

The application takes a canonical historical price dataset and produces validated asset returns, portfolio P&L, VaR/ES estimates, backtest diagnostics, stress losses, and model-comparison outputs. The Streamlit terminal provides interactive analysis, while the FastAPI layer exposes selected workflows programmatically.

## Key capabilities

- **Portfolio analytics:** Explicit asset selection, weight validation, portfolio returns, P&L, value evolution, volatility, cumulative return, and drawdown.
- **Risk models:** Historical, Gaussian, Student-t, empirical-bootstrap Monte Carlo, and GARCH(1,1) VaR and Expected Shortfall.
- **VaR backtesting:** Out-of-sample forecasts with violation rates, Kupiec unconditional-coverage testing, Christoffersen independence testing, and conditional coverage.
- **Champion–Challenger evaluation:** Configurable eligibility and scoring across coverage validity, exception-rate accuracy, stability where forecast data is available, and operational cost.
- **Stress testing:** Uniform, asset-specific, historical replay, volatility-scaled, and deterministic joint scenarios with asset-level loss attribution.
- **ML risk diagnostics:** Naive rolling volatility, Ridge, Random Forest, and XGBoost recent-period diagnostics. These are clearly separated from the standard full-period validation framework.
- **Professional interfaces:** A dark Streamlit risk terminal, local FastAPI endpoints, and an optional C++17 numerical-computing experiment.

## Live dashboard

The deployed dashboard presents the project as a single risk workflow rather than a collection of independent models.

The main workspaces are:

| Workspace | Purpose |
|---|---|
| Overview | Portfolio value, returns, volatility, drawdown, allocation, and reference-model context |
| Portfolio | Data-driven asset selection and explicitly validated allocation weights |
| Risk | VaR and ES comparison across the supported models |
| Stress Testing | Scenario losses, stressed values, and asset-level attribution |
| Backtesting | Persisted Phase 4 reference-portfolio validation results |
| Model Diagnostics | Champion–Challenger results, GARCH diagnostics, and recent-period ML diagnostics |

The default demonstration portfolio is:

```text
HDFCBANK.NS   50%
ICICIBANK.NS  30%
RELIANCE.NS   20%
```

This is only a reference allocation. The dashboard discovers all valid assets from the canonical project data and does not silently normalize invalid user weights.

## Architecture

```text
Raw prices
    ↓
Data Engine: loading, schema validation, dates, missingness, price checks
    ↓
Validated asset returns
    ↓
Portfolio & P&L Engine
    ├── Portfolio returns, value, P&L, and loss
    ├── Risk Engine: VaR and Expected Shortfall
    │       └── Historical · Gaussian · Student-t · Monte Carlo · GARCH
    ├── Backtesting Engine
    │       └── Coverage tests and exception diagnostics
    ├── Champion–Challenger Evaluator
    ├── Stress Testing Engine
    └── ML Risk Diagnostics
            ↓
    Streamlit Dashboard / FastAPI API
```

The core engines are independent of the presentation layer. The dashboard consumes existing portfolio, risk, stress, and persisted validation outputs; it does not duplicate the mathematical implementations or rerun the expensive full historical backtest when the user changes the interactive portfolio.

## Quantitative conventions

Risk uses a loss-based sign convention:

```text
P&L  = stressed or realized value − beginning value
Loss = −P&L
```

Positive loss represents an adverse outcome. VaR is the right-tail quantile of the loss distribution, and Expected Shortfall measures the average loss in the selected right tail. Monetary and relative units are kept explicit throughout the risk layer.

The stored full-period Champion–Challenger evaluation selects GARCH under the configured eligibility and scoring rules. This is a result of the documented validation framework, not a universal claim that GARCH is the best model in every setting.

## Models and validation

| Area | Implementation |
|---|---|
| Historical risk | Empirical loss quantiles and fractional-tail Expected Shortfall |
| Gaussian risk | Normal parametric VaR/ES using estimated loss mean and sample volatility |
| Student-t risk | SciPy maximum-likelihood location, scale, and degrees-of-freedom fit |
| Monte Carlo | Seeded empirical bootstrap of observed portfolio losses |
| GARCH | Gaussian GARCH(1,1), fitted with `arch`, with one-step conditional forecasting |
| Backtesting | Rolling/expanding out-of-sample forecasts with coverage diagnostics |
| Model selection | Configurable Champion–Challenger eligibility and weighted scoring |

Detailed mathematical methodology is documented in [`docs/risk_engine.md`](docs/risk_engine.md), [`docs/backtesting.md`](docs/backtesting.md), and [`docs/champion_challenger.md`](docs/champion_challenger.md).

## Data and reproducibility

The supplied source dataset is:

```text
data/raw/dataset_prices.csv
```

It is a wide-format historical price dataset containing a date field and 50 equity ticker columns, covering 2008-01-02 through 2024-12-30. The Phase 1 data engine produces canonical files under `data/processed/`, including validated prices, decimal daily returns, and asset metadata.

Historical observations are treated as immutable. The system validates, reports, and structurally organizes data without fabricating prices, imputing genuine missing observations, smoothing values, or removing extreme financial events to improve model performance.

Persisted Phase 4 and Phase 5 summaries are used by the dashboard for the reference portfolio. New portfolio allocations can be analyzed interactively, but comparable historical backtesting must be run separately through the backtesting workflow.

## Technology stack

| Technology | Role |
|---|---|
| Python 3.11+ | Primary application and quantitative-computing language |
| NumPy / pandas | Numerical arrays, tabular data, and portfolio calculations |
| SciPy | Statistical distributions and Student-t estimation |
| `arch` | GARCH estimation and forecasting |
| scikit-learn / XGBoost | Recent-period ML diagnostics |
| Streamlit | Interactive risk dashboard |
| FastAPI / Uvicorn | Local programmatic API |
| pytest / Ruff | Testing, formatting, and linting |
| C++17 / MinGW GCC | Optional standalone numerical performance experiment |

No external APIs, credentials, databases, or cloud services are required for the application’s core workflows.

## Repository structure

```text
RiskChallenger/
├── api/                         # FastAPI application
├── dashboard/                   # Streamlit dashboard and data access
├── cpp/                         # Optional C++ implementation and benchmark
├── data/
│   ├── raw/                     # Original supplied dataset
│   ├── processed/               # Canonical prices, returns, metadata
│   └── reports/                 # Persisted validation summaries
├── docs/                        # Component methodology and API documentation
├── src/riskchallenger/
│   ├── data/                    # Data loading and validation
│   ├── portfolio/               # Portfolio and P&L engine
│   ├── risk/                    # VaR, ES, Monte Carlo, and GARCH models
│   ├── backtesting/             # VaR forecasting and coverage tests
│   ├── champion/                # Champion–Challenger evaluation
│   ├── stress/                  # Scenario-based stress testing
│   └── ml/                      # Recent-period ML diagnostics
├── tests/                       # Python test suite
├── pyproject.toml               # Project and dependency configuration
└── README.md
```

## Installation

Windows PowerShell instructions:

```powershell
git clone https://github.com/Yashu0508/RiskChallenger.git
Set-Location RiskChallenger

py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks environment activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
```

## Run locally

### Streamlit dashboard

```powershell
streamlit run dashboard\\app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`.

### FastAPI service

In a second terminal with the virtual environment activated:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Useful local endpoints:

```text
API base:  http://127.0.0.1:8000
Swagger:   http://127.0.0.1:8000/docs
Health:    http://127.0.0.1:8000/health
```

The API exposes portfolio analysis, VaR/ES, model comparison, stress testing, Champion information, and ML diagnostic routes. See [`docs/api.md`](docs/api.md) for request schemas.

### Tests and quality checks

```powershell
python -m pytest
ruff format --check .
ruff check .
python -m compileall src dashboard api
```

### Optional C++ layer

The C++ component is not required by the dashboard or API. Build and benchmark instructions are available in [`cpp/README.md`](cpp/README.md). It implements a focused weighted portfolio-return calculation and is retained as an optional performance-engineering demonstration. The measured standalone process bridge was slower than NumPy for the documented benchmark workload because of startup and serialization overhead; no unsupported speedup claim is made.

## Example workflow

1. Load or validate the supplied canonical market data.
2. Start the dashboard and select assets from the data-derived universe.
3. Enter portfolio weights that sum to one and set the initial portfolio value.
4. Review portfolio returns, value evolution, volatility, and drawdown.
5. Compare loss-based VaR and ES across the supported risk models.
6. Apply stress scenarios and inspect asset-level loss attribution.
7. Review the persisted reference backtest and Champion–Challenger result.
8. Inspect recent-period ML diagnostics separately from standard model validation.
9. Use the local API for programmatic analysis when required.

## Engineering principles

- Separate data, portfolio, risk, validation, stress, diagnostics, and presentation concerns.
- Preserve historical financial observations and report data-quality issues transparently.
- Validate weights, dates, prices, returns, scenarios, and model inputs explicitly.
- Avoid silent imputation, weight normalization, look-ahead bias, and accidental recalculation of persisted backtests.
- Prefer deterministic, reproducible calculations where stochastic methods are used.
- Keep the optional C++ component small and independent from the primary Python application.

## Limitations

- Results depend on the supplied historical price dataset and each model’s assumptions.
- The dashboard’s stored backtesting and Champion–Challenger summaries refer to the validated reference portfolio.
- Stress scenarios are explicit adverse scenarios, not probability forecasts.
- ML outputs are recent-period diagnostics and are not a replacement for full-period risk-model validation.
- The optional C++ benchmark represents one environment and workload; its process bridge is not automatically faster than NumPy.
- Price-based analysis does not capture every real-world consideration, including liquidity, transaction costs, dividends, and market impact.

## Disclaimer

RiskChallenger is an educational and engineering project. It is not investment advice, a trading recommendation, or a guarantee of future performance. Users should independently validate data, assumptions, and model suitability before relying on any output for a real financial decision.
