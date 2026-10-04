"""One-step-ahead Gaussian GARCH(1,1) loss risk forecasts."""

import warnings

import numpy as np
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning
from scipy.stats import norm

from riskchallenger.logging import get_logger
from riskchallenger.risk.base import RiskModel
from riskchallenger.risk.result import RiskUnit
from riskchallenger.risk.validation import validate_unit

LOGGER = get_logger("risk.garch")


class GARCHRiskModel(RiskModel):
    """GARCH(1,1) conditional Gaussian risk for Phase 2 portfolio returns.

    Returns are supplied in decimal units and scaled to percentage points for
    estimation. Monetary risk requires the caller to supply the portfolio value
    explicitly; relative risk does not.
    """

    model_name = "garch_1_1_gaussian"
    minimum_sample_size = 100

    def __init__(self) -> None:
        super().__init__()
        self._portfolio_value: float | None = None
        self._fit_result = None
        self._constant_return: float | None = None

    def fit(
        self,
        returns: object,
        *,
        unit: RiskUnit | str,
        portfolio_value: float | None = None,
    ) -> "GARCHRiskModel":
        """Fit GARCH to portfolio returns, requiring value for monetary risk."""
        parsed_unit = validate_unit(unit)
        if parsed_unit is RiskUnit.MONETARY:
            if isinstance(portfolio_value, bool) or portfolio_value is None:
                raise ValueError(
                    "portfolio_value must be explicitly supplied and positive "
                    "for monetary GARCH risk"
                )
            try:
                value = float(portfolio_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "portfolio_value must be finite and positive"
                ) from error
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("portfolio_value must be finite and positive")
        else:
            if portfolio_value is not None:
                raise ValueError(
                    "portfolio_value must be omitted for relative GARCH risk"
                )
            value = None

        super().fit(returns, unit=parsed_unit)
        self._portfolio_value = value
        self._fit_result = None
        self._constant_return = None
        observations = self._losses
        if observations is None:
            raise RuntimeError("GARCH input was not stored")
        if np.all(observations == observations[0]):
            self._constant_return = float(observations[0])
            LOGGER.warning(
                "GARCH input is constant; using the deterministic zero-volatility limit"
            )
            return self

        scaled_returns = observations * 100.0
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model = arch_model(
                    scaled_returns,
                    mean="Constant",
                    vol="GARCH",
                    p=1,
                    o=0,
                    q=1,
                    dist="normal",
                    rescale=False,
                )
                fitted = model.fit(disp="off", show_warning=False, update_freq=0)
        except Exception as error:
            raise RuntimeError(f"GARCH(1,1) fitting failed: {error}") from error

        warning_messages = [
            f"{warning.category.__name__}: {warning.message}"
            for warning in caught_warnings
        ]
        has_convergence_warning = any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught_warnings
        )
        convergence_flag = int(getattr(fitted, "convergence_flag", -1))
        if has_convergence_warning or convergence_flag != 0:
            detail = (
                "; ".join(warning_messages) or "optimizer convergence flag is non-zero"
            )
            raise RuntimeError(
                f"GARCH(1,1) fit did not converge (flag={convergence_flag}): {detail}"
            )
        self._fit_result = fitted
        self._fit_warnings = warning_messages
        LOGGER.info(
            "Fitted Gaussian GARCH(1,1) to %d portfolio return observations",
            observations.size,
        )
        return self

    def _calculate(
        self, confidence_level: float, returns: np.ndarray
    ) -> tuple[float, float, dict[str, float | int | str | None]]:
        if self._constant_return is not None:
            forecast_mean = self._constant_return
            forecast_volatility = 0.0
            parameters: dict[str, float | int | str | None] = {
                "fit_status": "constant-return deterministic limit",
                "mean_return": forecast_mean,
                "forecast_volatility": forecast_volatility,
                "innovation_distribution": "Gaussian",
                "garch_specification": "GARCH(1,1)",
                "fit_warnings": "",
            }
        else:
            if self._fit_result is None:
                raise RuntimeError("GARCH model must be fitted before calculation")
            fitted = self._fit_result
            fitted_parameters = fitted.params
            omega = float(fitted_parameters["omega"]) / 10_000.0
            alpha = float(fitted_parameters["alpha[1]"])
            beta = float(fitted_parameters["beta[1]"])
            if (
                not np.isfinite([omega, alpha, beta]).all()
                or omega <= 0
                or alpha < 0
                or beta < 0
            ):
                raise RuntimeError(
                    "Fitted GARCH parameters violate omega > 0, alpha >= 0, beta >= 0"
                )
            try:
                forecast = fitted.forecast(horizon=1, method="analytic", reindex=False)
                forecast_mean = float(forecast.mean.iloc[-1, 0]) / 100.0
                variance_percent_squared = float(forecast.variance.iloc[-1, 0])
                forecast_volatility = float(np.sqrt(variance_percent_squared) / 100.0)
            except Exception as error:
                raise RuntimeError(
                    f"GARCH one-step forecast failed: {error}"
                ) from error
            if not np.isfinite([forecast_mean, forecast_volatility]).all():
                raise RuntimeError("GARCH forecast contains non-finite values")
            persistence = alpha + beta
            fit_warnings = list(getattr(self, "_fit_warnings", []))
            if persistence >= 1.0:
                fit_warnings.append(
                    "alpha + beta >= 1; the fitted process has no finite "
                    "unconditional variance"
                )
            parameters = {
                "omega": omega,
                "alpha_1": alpha,
                "beta_1": beta,
                "alpha_plus_beta": persistence,
                "mean_return": forecast_mean,
                "forecast_volatility": forecast_volatility,
                "innovation_distribution": "Gaussian",
                "garch_specification": "GARCH(1,1)",
                "fit_method": "arch maximum likelihood",
                "convergence_flag": int(fitted.convergence_flag),
                "fit_warnings": "; ".join(fit_warnings),
            }

        z_score = float(norm.ppf(confidence_level))
        relative_var = -forecast_mean + forecast_volatility * z_score
        relative_es = -forecast_mean + forecast_volatility * float(
            norm.pdf(z_score)
        ) / (1.0 - confidence_level)
        multiplier = self._portfolio_value if self._unit is RiskUnit.MONETARY else 1.0
        if multiplier is None:
            raise RuntimeError("Monetary GARCH risk is missing portfolio_value")
        var = relative_var * multiplier
        expected_shortfall = relative_es * multiplier
        parameters["forecast_volatility"] = forecast_volatility
        parameters["portfolio_value_multiplier"] = multiplier
        return var, expected_shortfall, parameters
