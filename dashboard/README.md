# RiskChallenger dashboard

The Streamlit application provides a local dark terminal interface for the
validated portfolio, risk, stress, backtesting, and model-diagnostic layers.
Asset choices are discovered from the canonical price and return datasets;
the three-stock portfolio remains the initial demonstration allocation.

Install the project dependencies from the repository root, then run:

```bash
streamlit run dashboard/app.py
```

The app reads local canonical data and persisted result summaries. It makes no
external requests and never reruns the historical backtest. Weight totals are
validated without normalization. Detailed methodology and architecture remain
in the project-level `docs/` directory.
