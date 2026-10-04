# Scenario-Based Stress Testing

## Purpose and separation from VaR

VaR estimates a loss quantile at a specified probability level. Stress testing
asks what the portfolio would do under a specified adverse shock. Stress
scenarios in RiskChallenger are deterministic and do not assign probabilities
to scenarios or turn them into another VaR estimate. The stress engine is also
separate from the Phase 5 Champion–Challenger framework: a scenario result does
not change the backtested forecasting-model Champion.

## Portfolio and valuation convention

The engine consumes the Phase 2 `Portfolio` asset order, weights, and initial
value (or a caller-supplied current base value). For asset simple-return shocks
`s_i` and beginning scenario value `V`,

```text
stressed portfolio return = sum_i(weight_i * s_i)
P&L_i                    = V * weight_i * s_i
P&L                       = sum_i(P&L_i)
stressed value            = V + P&L
loss                      = -P&L
loss percentage           = 100 * loss / V
```

Thus positive loss is adverse and negative loss is a gain. Asset contributions
are additive and reconcile to portfolio P&L. Weights are held fixed for this
one-period mark-to-market; there is no rebalancing, transaction cost, market
impact, or path-dependent trading.

Each result includes scenario identity/type, base and stressed values, P&L,
loss, loss percentage, asset attribution, assumptions, and optional event date
bounds. `loss_percentage` is in percentage points (e.g. `5.0` means 5%).

## Scenario definitions

### Uniform market shock

Applies one supplied simple return to every selected asset. For example,
`-0.10` means a 10% price decline for each asset. The shock is an explicit
scenario input, not a fitted or probabilistic estimate.

### Asset-specific shock

Applies an explicit return to every portfolio asset. The supplied asset set
must match the portfolio exactly; unknown, missing, or duplicated asset names
are rejected rather than assumed to have zero shock.

### Historical replay

Can replay a directly supplied asset-return vector, or derive event-window
cumulative simple returns from the canonical price dataset. For price-window
construction, the event base is the last available close strictly before the
requested start date, and the endpoint is the last available close on or before
the requested end date:

```text
event return_i = endpoint price_i / base price_i - 1
```

Today's portfolio weights are applied to these asset returns. Historical
portfolio weights are not reused. Endpoint prices must exist, be finite, and
be positive. The raw and canonical price files are read-only inputs.

The supplied dataset spans 2008-01-02 through 2024-12-30 and contains selected
asset prices covering both defined event windows used in this phase:

| Scenario window | Start | End |
|---|---|---|
| Global Financial Crisis stress window | 2008-09-15 | 2009-03-09 |
| COVID-19 crash stress window | 2020-02-19 | 2020-03-23 |

These are explicitly chosen event windows, not claims that a single interval
captures every consequence of either crisis.

### Volatility-scaled realization

Multiplying volatility alone does not uniquely specify a return. Therefore a
`VolatilityShock` also requires an explicit reference return vector. It
deterministically applies `stressed_return_i = multiplier * reference_return_i`
with the realization centered at zero. The multiplier must be at least one.
This is a scaled scenario realization, not a volatility forecast, random
simulation, or calibrated conditional distribution.

### Deterministic joint shock

Each asset's shock is constructed as:

```text
asset shock_i = common market shock + idiosyncratic shock_i
```

The common component creates deterministic joint co-movement. Any omitted
idiosyncratic term is explicitly treated as zero. This simple factor-style
construction is not a stochastic correlation model and does not estimate a
correlation matrix.

## Validation and integrity

Scenario names must be nonempty. Shocks must be numeric, finite simple returns
strictly greater than `-1`, so each shocked asset retains a positive price.
Asset-specific and historical vectors must cover every selected asset exactly
once. Volatility multipliers must be finite and at least one. Historical date
bounds must be valid, timezone-naive, and ordered; necessary price endpoints
must be present. Base portfolio value and resulting total stressed value must
be finite and positive. Invalid inputs fail loudly; no shocks, missing values,
or prices are repaired or imputed.

## Asset attribution

For each selected asset, the result reports weight, return shock, starting
exposure, stressed exposure, P&L contribution, and loss contribution:

```text
starting exposure_i = V * weight_i
P&L contribution_i  = starting exposure_i * shock_i
loss contribution_i = -P&L contribution_i
```

Summing asset P&L contributions reproduces portfolio P&L (up to floating-point
precision). No attempt is made to allocate interaction terms because the
specified one-period weighted-simple-return portfolio formula is additive.

## Limitations

Stress results depend on the chosen scenario, current weights/value, and the
price-only dataset. Historical replay is not a forecast or probability
estimate, and price returns may not capture dividends or liquidity effects.
Uniform, asset-specific, and deterministic joint shocks do not model market
microstructure, rebalancing, nonlinear derivatives, or feedback effects.
Volatility scaling inherits the supplied reference return vector and is only a
transparent deterministic sensitivity. Stress outputs complement—not replace—
VaR/ES estimates, backtesting, or the Champion–Challenger evaluation.
