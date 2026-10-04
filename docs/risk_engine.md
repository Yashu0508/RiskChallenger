# Core risk engine

Risk models consume losses supplied by the Phase 2 portfolio engine. The
convention is `loss = -P&L`: positive values are losses and negative values
are gains. A caller must label observations as `relative` (decimal return-loss
units) or `monetary` (currency units); the risk package does not infer or
convert units. Confidence `alpha` is strictly between zero and one and
represents the right-tail loss quantile, so `VaR_alpha = Q_alpha(loss)`.

## Historical model

Historical VaR uses NumPy's `method="linear"` empirical quantile. Historical
ES integrates the empirical quantile function over the upper `1 - alpha`
fraction of the sample. It uses the exact tail observation mass, including a
fractional weight on the boundary observation when the mass is non-integer.
This avoids changing tail probability due to rounding the number of tail rows.
It can differ from a plain arithmetic average of all observations satisfying
`loss >= VaR` when ties occur. The reported VaR and ES both use the same
positive-loss right-tail convention.

## Gaussian model

Loss location is the arithmetic sample mean and scale is the sample standard
deviation (`ddof=1`). For `z = Phi^-1(alpha)`, VaR is `mu + sigma*z` and ES is
`mu + sigma*phi(z)/(1-alpha)`. This assumes an adequately represented Gaussian
loss distribution. Constant samples are supported with zero estimated scale.

## Student-t model

The model fits SciPy's `scipy.stats.t` distribution by maximum likelihood,
estimating `df`, `loc`, and `scale`. SciPy's scale parameter is not the
Student-t standard deviation; when `df > 2`, standard deviation is
`scale*sqrt(df/(df-2))`. For `q = t_df^-1(alpha)`, VaR is `loc + scale*q` and
ES is
`loc + scale * ((df + q^2)/(df - 1)) * pdf_df(q)/sf_df(q)`.
Finite ES requires `df > 1`; fits at or below that threshold fail clearly.
Constant samples are returned as a degenerate distribution with unidentified
degrees of freedom. At least three observations are required for a nonconstant
Student-t fit.

## Interface and sample requirements

Each model implements `fit(losses, unit=...)` and `calculate(alpha)`, which
returns a `RiskResult` containing VaR, ES, sample size, unit, convention, and
model parameters. Historical VaR/ES permits a single observation; Gaussian
requires at least two for sample standard deviation; Student-t requires at
least three. Missing, non-numeric, non-finite, and multidimensional input is
rejected without imputation.

## Monte Carlo empirical bootstrap

`MonteCarloRiskModel` samples observed portfolio losses with replacement using
a dedicated `numpy.random.default_rng`. Its defaults are 100,000 simulations and
seed 2026; both are configurable and validated. The same historical input,
simulation count, and seed reproduce the same result. No parametric distribution
is imposed. This one-period iid bootstrap preserves the empirical marginal
distribution but does not model serial dependence or changing volatility. At
least 20 historical observations are required. Simulated VaR uses the same
linear quantile and simulated ES the same fractional upper-tail method as the
historical model.

## Gaussian GARCH(1,1)

`GARCHRiskModel` consumes the Phase 2 portfolio return series in decimal units.
It estimates a constant-mean GARCH(1,1) model with Gaussian innovations using
the `arch` maximum-likelihood implementation. For residuals `epsilon_t`, the
variance recursion is `sigma_t^2 = omega + alpha*epsilon_(t-1)^2 +
beta*sigma_(t-1)^2`, with `omega > 0`, `alpha >= 0`, and `beta >= 0`. Returns are
scaled to percentage points for fitting; reported forecast mean and volatility
are converted back to decimals. At least 100 observations are required.

The one-step analytic forecast uses information through the final historical
return. Under Gaussian innovations, `R_(t+1) ~ N(mu_(t+1), sigma_(t+1)^2)`.
Relative loss is `-R_(t+1)`, giving `VaR_alpha = -mu + sigma*z_alpha` and
`ES_alpha = -mu + sigma*phi(z_alpha)/(1-alpha)`. For monetary risk both
estimates are multiplied by the explicitly supplied positive portfolio value;
the model never infers it. For a forecast made at the end of an observed sample,
the caller should supply the portfolio value at that forecast origin (for
example, the final marked-to-market value from Phase 2). Relative output needs
no portfolio value. A constant return series is handled as a deterministic
zero-volatility limit because its GARCH parameters cannot be identified.

Optimizer convergence warnings and non-zero convergence flags cause the fit to
fail instead of returning a potentially unreliable forecast. Other fitting
warnings are retained in result metadata. Historical estimates are unconditional
empirical distribution summaries, Monte Carlo estimates are simulated empirical
tail summaries, and GARCH estimates are conditional on the final observed
portfolio return and its fitted volatility state.
