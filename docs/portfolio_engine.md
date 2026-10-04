# Portfolio and P&L engine

`Portfolio` pairs an ordered sequence of unique asset names with a weight for
each asset and a finite positive initial portfolio value. Weight validation is
explicit: by default weights must be non-negative (`long_only=True`) and sum to
one (`fully_invested=True`, with an absolute tolerance of `1e-8`). The engine
never normalizes weights. Set either option to `False` only when the caller
intends that alternative convention.

`load_return_dataset()` reads the Phase 1 `data/processed/nse50_returns.csv`
without modifying it. `calculate_portfolio()` selects assets in the exact order
specified by the portfolio, then computes daily portfolio return as
`R_p,t = sum_i(w_i * r_i,t)`. Returns remain decimal values.

The Phase 1 leading row has no defined returns because no prior price exists.
When that first row is missing for every selected asset, the engine excludes
that row from portfolio output. Partial initial missingness and any missing or
non-finite value on later dates raise an error. No return imputation is done.

For each date, P&L uses the value immediately before that day's return:
`P&L_t = V_(t-1) * R_p,t`. Loss is defined as `L_t = -P&L_t`, so gains have
negative loss. Post-return portfolio value is `V_t = V_(t-1) * (1 + R_p,t)`;
the first output value follows the first defined portfolio return applied to
the configured initial value.

The result exposes date-indexed portfolio returns, values, P&L, and losses, plus
basic descriptive statistics. Mean and extrema use daily decimal returns;
standard deviation uses the sample convention (`ddof=1`). Cumulative return is
compounded and total P&L is the sum of daily P&L. The portfolio is held at
constant weights for each daily return calculation; transaction costs,
rebalancing mechanics, and trading constraints are outside this phase.
