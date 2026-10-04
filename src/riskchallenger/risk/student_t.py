"""Fitted Student-t loss VaR and Expected Shortfall."""

import numpy as np
from scipy.stats import t

from riskchallenger.risk.base import RiskModel


class StudentTRiskModel(RiskModel):
    """Student-t loss model fitted by SciPy maximum likelihood.

    SciPy's parameterization is used: ``t(df, loc, scale)`` where ``scale`` is
    a distribution scale parameter, not standard deviation. ES is finite only
    when fitted ``df > 1``.
    """

    model_name = "student_t"
    minimum_sample_size = 3

    def _calculate(
        self, confidence_level: float, losses: np.ndarray
    ) -> tuple[float, float, dict[str, float | int | str | None]]:
        if np.all(losses == losses[0]):
            constant = float(losses[0])
            return (
                constant,
                constant,
                {
                    "degrees_of_freedom": None,
                    "location": constant,
                    "scale": 0.0,
                    "fit_method": (
                        "degenerate constant sample; degrees of freedom unidentified"
                    ),
                },
            )

        try:
            degrees_of_freedom, location, scale = t.fit(losses)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            raise ValueError(
                f"Student-t maximum-likelihood fit failed: {error}"
            ) from error
        parameters = (degrees_of_freedom, location, scale)
        if (
            not np.isfinite(parameters).all()
            or scale <= 0.0
            or degrees_of_freedom <= 1.0
        ):
            raise ValueError(
                "Fitted Student-t parameters must be finite, scale > 0, and "
                "degrees_of_freedom > 1 for finite Expected Shortfall; got "
                f"df={degrees_of_freedom}, scale={scale}"
            )

        quantile = float(t.ppf(confidence_level, degrees_of_freedom))
        density = float(t.pdf(quantile, degrees_of_freedom))
        tail_probability = float(t.sf(quantile, degrees_of_freedom))
        value_at_risk = float(location + scale * quantile)
        tail_expectation = (
            (degrees_of_freedom + quantile**2)
            / (degrees_of_freedom - 1.0)
            * density
            / tail_probability
        )
        expected_shortfall = float(location + scale * tail_expectation)
        return (
            value_at_risk,
            expected_shortfall,
            {
                "degrees_of_freedom": float(degrees_of_freedom),
                "location": float(location),
                "scale": float(scale),
                "fit_method": "SciPy t.fit maximum likelihood",
                "scale_definition": (
                    "SciPy Student-t scale parameter, not standard deviation"
                ),
                "expected_shortfall_formula": (
                    "loc + scale * ((df + q^2)/(df - 1)) * pdf(q)/sf(q)"
                ),
            },
        )
