# RiskChallenger

**AI-Assisted Quantitative Risk Model Validation & Challenger Framework**

RiskChallenger is a local, modular portfolio risk-management platform for NSE equities. It turns historical prices into portfolio analytics, loss-based risk estimates, backtests, model diagnostics, and scenario-based stress results through a Python application layer.

## Key capabilities

- Historical, Gaussian, Student-t, Monte Carlo, and GARCH Value at Risk (VaR) and Expected Shortfall (ES)
- Out-of-sample VaR backtesting with exception-rate, Kupiec, Christoffersen, and conditional-coverage diagnostics
- Configurable Champion–Challenger evaluation based on forecast validation and operational criteria
- Deterministic portfolio stress scenarios, including historical replay and asset-level attribution
- Recent-period diagnostic challengers using naive rolling volatility, Ridge, Random Forest, and XGBoost
- Interactive dark Streamlit terminal and a local FastAPI interface
- Optional C++17 portfolio-return calculation experiment, benchmarked against NumPy

The dashboard and API operate locally using project data; no external services or credentials are required. Supplied historical price observations are treated as read-only. The three-stock portfolio shown by default is a demonstration, not the complete asset universe.

## Quantitative conventions

Risk estimates use a loss-based convention: positive loss is adverse, and VaR is the right-tail quantile of losses. The Phase 2 portfolio engine calculates weighted portfolio returns, beginning-of-period-value P&L, and loss consistently. Detailed formulas, estimation assumptions, and validation methodology are documented in [`docs/`](docs/).

Backtesting results and the Champion selection are distinct from the recent-period ML diagnostic. The stored full-period reference evaluation selects GARCH under the configured Champion–Challenger rules; this is a validation outcome, not a claim that GARCH is universally best. The dashboard displays persisted backtest summaries and does not rerun the expensive full historical backtest when portfolio selections change.

## Architecture

```text
Raw prices → Data Engine → Validated asset returns → Portfolio & P&L
                                                       ├→ Risk models → Backtesting → Champion–Challenger
                                                       ├→ Scenario stress testing
                                                       └→ Recent-period ML diagnostics
                                                                  ↓
                                                Streamlit dashboard / FastAPI

Optional C++17 calculation ← Python adapter (not required by the application)
```

Python is the primary application and quantitative-computing layer. Data processing, portfolio calculations, risk models, backtesting, model evaluation, and stress testing are separated into reusable modules. The dashboard and API present these capabilities without replacing their core calculations.

## Technology

Python 3.11+, NumPy, pandas, SciPy, `arch`, scikit-learn, XGBoost, Streamlit, FastAPI, Uvicorn, pytest, and Ruff. An optional standalone C++17 executable demonstrates a weighted portfolio-return calculation; it is not needed to run the dashboard or API.

## Dataset

The supplied raw dataset is `data/raw/dataset_prices.csv`. It is a wide-format historical price file with a date field and 50 equity ticker columns, spanning 2008-01-02 through 2024-12-30. Phase 1 creates canonical price, return, and metadata files under `data/processed/`; the dashboard discovers its selectable asset universe from project data rather than a manually maintained ticker list. No additional market data source is required.

## Project structure

```text
Risk_Manager_System/
├── api/                         # Local FastAPI application
├── cpp/                         # Optional C++ implementation, tests, benchmark
├── dashboard/                   # Streamlit application
├── data/
│   ├── raw/                     # Supplied source data
│   ├── processed/               # Canonical prices, returns, metadata
│   └── reports/                 # Persisted diagnostics and reports
├── docs/                        # Methodology and component guides
├── src/riskchallenger/
│   ├── data/                    # Data loading and validation
│   ├── portfolio/               # Portfolio returns and P&L
│   ├── risk/                    # VaR and ES models
│   ├── backtesting/             # Forecast evaluation
│   ├── champion/                # Champion–Challenger framework
│   ├── stress/                  # Scenario analysis
│   └── ml/                      # Recent-period risk diagnostics
├── tests/
└── pyproject.toml
```

## Installation

In Windows PowerShell, open the repository directory and create an isolated environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install includes the project runtime dependencies and the optional development tools used for testing and linting.

## Run the project

### Dashboard

```powershell
streamlit run dashboard/app.py
```

Streamlit prints the local address, usually `http://localhost:8501`. The dashboard supports portfolio construction from the available dataset assets, risk and stress views, and persisted backtesting and model-diagnostic summaries.

### API

In a separate terminal with the virtual environment activated:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

The API base address is `http://127.0.0.1:8000`; interactive Swagger documentation is at [`/docs`](http://127.0.0.1:8000/docs), and the health endpoint is [`/health`](http://127.0.0.1:8000/health). Available route groups cover portfolio analysis, VaR/ES, risk comparison, stress scenarios, Champion information, and ML diagnostics. See [`docs/api.md`](docs/api.md) for request details.

### Tests

```powershell
python -m pytest
```

### Optional C++ calculation

The C++17 component can be built and tested independently; instructions are in [`cpp/README.md`](cpp/README.md). A controlled 200,000-row, 50-asset benchmark measured the standalone C++ process bridge slower than NumPy for this workload. It is retained as an optional performance-engineering experiment—not as an automatic acceleration or application dependency. See [`docs/cpp_performance.md`](docs/cpp_performance.md) for methodology and results.

## Example workflow

1. Install the project and run the Phase 1 data pipeline for the supplied raw prices.
2. Start the dashboard and select assets from the data-derived universe; enter weights that sum to one and set portfolio value.
3. Review portfolio return and value analytics, then inspect loss-based VaR and ES across the supported risk models.
4. Explore stress scenarios and their asset-level P&L attribution.
5. Review the persisted reference backtest and Champion–Challenger result separately from the recent-period ML diagnostic.
6. Use the local API for programmatic access, or build the optional C++ calculation to reproduce its standalone benchmark.

## Engineering principles

- Separate data, portfolio, risk, validation, stress, and presentation responsibilities.
- Preserve historical observations; report data-quality issues rather than fabricating or silently repairing financial values.
- Validate portfolio weights and scenario inputs explicitly; do not silently normalize weights or impute returns.
- Keep backtest forecasts out of sample and separate persisted reference results from on-demand portfolio analytics.
- Make stochastic calculations reproducible where applicable and keep optional components optional.

## Limitations

Results are limited to the supplied historical dataset and the assumptions of each model. Backtest summaries describe a stored reference-portfolio run; a new portfolio requires a separately run backtesting pipeline for comparable historical validation. ML outputs are recent-period diagnostics, not a replacement for the core risk models or a guarantee of future performance. Stress tests are defined scenarios, not probability forecasts. The C++ benchmark reflects one machine and workload, and its process/serialization overhead exceeded the NumPy runtime in that measurement.

## Disclaimer

RiskChallenger is an educational and engineering project. Its outputs are not investment advice, a trading recommendation, or a guarantee of future results. Validate assumptions and data suitability before using any output for a real financial decision.
