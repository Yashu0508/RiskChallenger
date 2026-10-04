# Champion–Challenger Framework

## Purpose and input

The evaluator consumes Phase 4 `BacktestRunResult` objects. It does not refit
risk models, calculate portfolio P&L, or rerun the backtester. Model-neutral
criteria select a Champion from eligible models; the other eligible models are
Challengers and models failing the eligibility gate remain separately visible
as ineligible. If no model is eligible, no Champion is declared.

## Default score

Each component lies in `[0, 1]`. The default weights are configurable through
`ScoreWeights` and sum to one:

| Component | Default weight |
|---|---:|
| Coverage validity | 0.40 |
| Exception-rate accuracy | 0.25 |
| Stability | 0.20 |
| Operational cost | 0.15 |

The composite is the weighted sum of available component scores. If a result
does not contain forecast rows, stability cannot be measured: it is marked
unavailable and the remaining component weights are renormalized. The missing
component is never silently assigned a favorable score.

### Coverage

For Kupiec unconditional coverage, Christoffersen independence, and
Christoffersen conditional coverage, a test passes when its p-value is at
least the configurable significance level (default 0.05). The coverage score
is the number of passing tests divided by three. P-values are not ranked by
magnitude: p=0.06 and p=0.90 both count as a pass.

### Exception-rate accuracy

With expected exception rate `p0 = 1 - confidence`, and observed exception
rate `p_hat`, define `error = abs(p_hat - p0)`. The score is

```text
max(0, 1 - error / p0)
```

Exact calibration scores one; an error at least as large as `p0` scores zero.
The rule depends on confidence and not model identity.

### Stability

Forecast rows are kept in date order and split into three contiguous,
approximately equal-count periods named early, middle, and recent. The first
period receives any remainder rows. At least 90 forecasts (30 per period) are
required to score stability; shorter forecast histories report it as
unavailable. Each period reports count, violations, violation rate, expected
rate, and Kupiec, independence, and conditional-coverage p-values.

For subperiod exception rates `p_hat_j` and conditional-coverage pass
indicators `c_j`, define:

```text
pass_fraction = mean(c_j)
accuracy_j = max(0, 1 - abs(p_hat_j - p0) / p0)
accuracy = mean(accuracy_j)
variability = min(1, population_std(p_hat_j) / p0)
stability_score = (pass_fraction + accuracy + (1 - variability)) / 3
```

This equally combines subperiod test validity, calibration accuracy, and
dispersion of exception rates. The test statistics are asymptotic and can be
weak in subperiods of limited length.

### Operational cost

Runtime is scored as

```text
1 / (1 + runtime_seconds / operational_time_scale_seconds)
```

The default scale is 60 seconds and is configurable. This bounded,
identity-independent transform makes runtime a secondary component rather than
a hard limit or eligibility gate.

## Eligibility and ranking

By default, a model is eligible only when its full-period conditional-coverage
test is not rejected at the configured significance level. This gate is
configurable. The Champion is the highest-scoring eligible model. All other
eligible models are Challengers; ineligible models are listed separately with
their failure reason. If there is no eligible model, the Champion is `None`.

Scores within the configured tie tolerance are resolved in this order:

1. Eligible/statistically valid coverage.
2. More of the three full-period coverage tests passing.
3. Smaller absolute exception-rate error.
4. Lower runtime.
5. Alphabetical model name.

## Methodological safeguards and limitations

Scoring weights, gates, normalization, period boundaries, and tie rules are
explicit. The rules are fixed independently of observed model identities and
do not encode a preferred model. Synthetic tests exercise different winners,
failed eligibility, cost tradeoffs, stability, and ties.

VaR magnitude is not a score: larger or smaller VaR is not inherently better.
ES is also not interchangeable with VaR; the Phase 4 run contains VaR
forecasts and does not provide out-of-sample ES forecasts for scoring. A model
that passes coverage is not thereby proven correct or superior. Results are
conditional on the test period, portfolio, confidence level, and Phase 4
backtest design.

The repository retains the Phase 4 summary statistics but not its full
per-date forecast frames. Those summary-only results support full-period
coverage, calibration, and runtime scoring, but cannot support actual-data
subperiod stability scores. The evaluator therefore reports stability as
unavailable and renormalizes the remaining weights in that evaluation. Future
backtest runs can pass their structured forecast frames directly to enable
early/middle/recent scoring without changing the evaluator.
