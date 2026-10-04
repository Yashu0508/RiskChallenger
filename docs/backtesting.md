# VaR Backtesting

## Purpose and conventions

The backtesting engine evaluates one-step-ahead VaR forecasts against the Phase
2 portfolio loss series. Loss is `-P&L`, so positive values are losses. At
confidence `alpha`, VaR is the right-tail loss quantile `Q_alpha(L)`. An
exception (violation) occurs only when the realized loss is strictly greater
than its forecast: `L_t > VaR_t`; equality is not an exception. The expected
exception probability is `1 - alpha`.

Backtests consume `PortfolioCalculationResult` and do not recalculate asset
returns, portfolio returns, P&L, or losses. They preserve the monetary unit of
the Phase 2 loss series.

## Forecasting design and no-lookahead rule

`BacktestingEngine` accepts model adapters implementing `fit(training_data)`
and `forecast_var(alpha)`. The adapter receives only observations with dates
strictly before the forecast date. The forecast is produced before the engine
reads that date's realized loss. Expanding windows are the default; rolling
windows warm up with available history and then use the configured fixed
trailing length. A configurable refit cadence reuses fitted parameters between
refits. Every model evaluates the same filtered forecast dates.

Adapters connect the existing Historical, Gaussian, Student-t, Monte Carlo,
and GARCH models without altering their risk conventions. The historical,
Gaussian, and Student-t adapters fit to prior monetary losses. Monte Carlo
bootstraps prior losses using a seeded generator (base seed plus refit number).
GARCH fits the prior portfolio returns and converts its relative forecast to
money using the portfolio value at the forecast origin, known before the
forecast return.

## Violation statistics

The summary reports the number of forecasts `T`, violations `x`, observed
exception rate `x/T`, expected exception rate `1-alpha`, and their difference.
These descriptive quantities do not by themselves establish model adequacy.

## Kupiec unconditional coverage test

Kupiec's likelihood-ratio test evaluates whether the unconditional exception
probability equals `p=1-alpha`. With observed rate `p_hat=x/T`,

```text
LR_uc = -2 log( ((1-p)^(T-x) p^x) /
                ((1-p_hat)^(T-x) p_hat^x) )
```

The statistic is compared with chi-square(1); the null is rejected when the
p-value is below the configured significance level. Boundary cases `x=0` and
`x=T` use the limiting likelihood, avoiding `0*log(0)` NaNs.

## Christoffersen independence test

Let `I_t` be the binary violation indicator and `N_ij` the count of adjacent
transitions from state `i` to `j`. Under independence, the probability of a
violation is common across prior states; under the alternative it may differ.
The likelihood-ratio statistic compares these restricted and unrestricted
Bernoulli transition likelihoods and uses chi-square(1). The report includes
all four transition counts. Constant sequences have no identified transition
probability contrast and return statistic zero (p-value one); this convention
does not imply that unconditional coverage is correct.

## Conditional coverage test

The conditional-coverage statistic is the sum of the Kupiec unconditional
coverage and Christoffersen independence statistics. It is compared with
chi-square(2), testing correct unconditional rate and independence jointly.
The reported reject/non-reject decision is a statistical test result, not a
model ranking or proof of validity.

## Scope and limitations

The engine supports date bounds, expanding/rolling windows, minimum initial
training size, refit frequency, and significance level. For the reproducible
actual run in this phase, the configuration uses alpha 0.95, 500 initial
observations, an expanding window, and daily refitting. Monte Carlo forecasts
use 10,000 bootstrap draws per refit, with base seed 2026 incremented by refit
number. Repeated refitting can be computationally expensive, especially for
Student-t and GARCH models. Coverage tests have limited power for short evaluation samples;
their chi-square calibration is asymptotic. The current tests are standard
one-step unconditional coverage, independence, and joint conditional coverage
tests; they do not constitute a broader validation or model-selection
framework.
