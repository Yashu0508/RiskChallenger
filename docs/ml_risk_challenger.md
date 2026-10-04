# Phase 7 — ML Risk Challenger

## Purpose and scope

This module forecasts next-session portfolio variance from information available
at the forecast origin. It is an additional forecasting challenger, not a
replacement for the established risk models or the Phase 5 evaluation rules.
The implementation does not include SHAP or any later dashboard/API work.

## Target and features

For portfolio simple return `r_t`, the supervised target indexed at time `t` is

```text
y_t = r_(t+1)^2
```

This is a noisy one-observation proxy for conditional variance, not an observed
latent variance. Features at `t` use only portfolio returns observed through
and including `t`: lagged returns (1, 2, 5, 10), sample rolling volatility
(5, 10, 20), rolling means (5, 20), 20-session downside volatility, largest
single-session loss, maximum drawdown, and 5-session compounded return. Rolling
windows use chronological observations; warm-up rows and the final unlabeled
target are excluded from supervised fitting, never imputed.

## Models

* Naive baseline: trailing 20-session sample variance, `ddof=1`.
* Ridge: standardized features and Ridge regression in a single scikit-learn
  pipeline; scaling parameters are fitted only on each training window.
* Random Forest: 100 trees, maximum depth 5, minimum leaf size 5, fixed seed 42.
* XGBoost: 100 trees, depth 3, learning rate 0.05, squared-error objective,
  fixed seed 42.

The tree-model defaults are intentionally fixed and modest. All supervised
models predict squared decimal returns; no price, asset-level, or future-date
features are used. A predicted variance below zero is projected to zero before
taking its square root, and the forecast metadata records whether this occurred.

## Refit and one-step VaR forecast

The backtester first applies its minimum training history and expanding/rolling
window selection. At a refit origin, supervised rows are constructed only from
that training history and require both complete features and an observed next
return target. The ML parameters are refitted every 20 forecast origins by
default. Between refits, features, the expanding historical mean return, and
beginning-of-period portfolio value refresh at every origin. Each date passed
to the adapter is checked to be strictly earlier than its forecast date.

For predicted standard deviation `sigma_(t+1)`, expanding historical mean
`mu_t`, confidence `alpha`, and `V_t` equal to the portfolio value immediately
before the forecast return, the monetary loss VaR is

```text
VaR_alpha,t+1 = -(mu_t + sigma_(t+1) * Phi^(-1)(1-alpha)) * V_t
```

This applies the established positive-loss/right-tail convention. It uses a
Gaussian one-step innovation to translate a variance forecast into VaR; it
does not provide an ML Expected Shortfall forecast. The mean is a simple
expanding arithmetic average, not a separately fitted mean model.

## Backtesting and refitting

ML adapters conform to the Phase 4 `VaRForecaster` interface. `refit_frequency`
can be set per model; for Phase 7 ML models it defaults to 20. The backtesting
engine supplies strictly prior returns, aligned dates, forecast date, and
beginning-of-period portfolio value, records model-supplied forecast metadata,
and continues to calculate the existing violation and coverage statistics.
No realized observation is used until after its forecast is formed.

For a common actual comparison, the ML challengers and slower static
distributional models use 20-origin refits; GARCH retains daily refitting
because its one-step conditional forecast must update every day. Refit
frequency and runtime are reported so these operational choices remain visible.

## Validation and interpretation limits

The models reject missing/non-finite returns, invalid date ordering, too few
labeled training examples, non-finite predictions, and malformed features.
There is no missing-data imputation. Scikit-learn scaling stays inside the
training pipeline. Fixed seeds and single-threaded tree fitting make the
estimator behavior reproducible in the supported environment.

Squared-return labels are noisy; the Gaussian conversion is a distributional
assumption and does not make the regression output a complete return model.
Prediction clipping can create zero-volatility forecasts in pathological
cases. Feature importance and Ridge coefficients are diagnostic descriptions,
not causal explanations. Results should be assessed using the common Phase 4
out-of-sample tests and Phase 5 evaluator, not by VaR magnitude alone.
