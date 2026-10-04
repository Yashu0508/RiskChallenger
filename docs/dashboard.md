# Phase 8 — Professional Risk Dashboard

## Purpose and boundaries

The Streamlit app is a presentation layer over existing RiskChallenger
components. Portfolio return/P&L, VaR/ES, and deterministic stress calculations
are delegated to Phase 2, Phase 3, and Phase 6 APIs. The UI does not implement
risk formulas, alter weights, or mutate canonical prices/returns.

## Local result snapshots

`data/reports/dashboard_phase4_summary.json` stores the previously validated
full-period Phase 4 aggregate outputs and Phase 5 summary-only score replay.
The historical run did not persist its daily forecast rows; therefore the app
shows violation and test summaries, not a fabricated timeline. Stability was
unavailable in the summary-only Champion evaluation and its weight was
renormalized according to the configured Phase 5 evaluator. The saved full-
period Champion is GARCH.

`data/reports/dashboard_phase7_diagnostic.json` stores the actual Phase 7
recent-period diagnostic outputs, including the separate 180-session period,
metrics, runtimes, and model interpretation values. It is clearly labeled as
a recent-period diagnostic and does not replace the full-period Phase 5
Champion. The dashboard reads the artifact instead of rerunning Phase 7 fits.

These compact JSON artifacts are generated from already validated project
results; they contain no underlying price history or redundant forecast panel.
The dashboard handles missing artifacts by displaying an availability notice.

## Views

- **Portfolio Overview:** user-configurable selected demo holdings, validated
  weights, value path, daily returns, annualized arithmetic summaries, and
  maximum drawdown.
- **VaR / ES & Backtesting:** existing Phase 3 historical, Gaussian,
  Student-t, Monte Carlo, and GARCH outputs, plus saved Phase 4 coverage
  results. The live risk estimates are labeled full-sample/in-sample.
- **Champion–Challenger:** saved Phase 5 Champion, component scores, eligibility,
  and the stability-data limitation.
- **Stress Testing:** uniform shocks, an asset-specific adverse case, and
  canonical-price Global Financial Crisis and COVID-19 replays, with Phase 6
  contribution attribution.
- **ML Risk Challengers:** saved naive/Ridge/Random Forest/XGBoost recent
  diagnostic with GARCH context, feature importance/coefficients, and an
  explicit caution that the result is not full-period model selection.

## Reproducibility and limitations

Run locally with `streamlit run dashboard/app.py`. No network access, API keys,
or external data are required. Interactive changes recalculate only the
selected portfolio-derived risk or stress view through existing APIs; the
historical backtest and saved ML diagnostic are not rerun. Annualization uses
252 sessions per year. Historical scenario replay applies current selected
weights to event-window cumulative simple returns from immutable canonical
prices.
