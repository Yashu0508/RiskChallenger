"""Gaussian parametric VaR and Expected Shortfall for losses."""

import numpy as np
from scipy.stats import norm

from riskchallenger.risk.base import RiskModel


class GaussianRiskModel(RiskModel):
    """Gaussian loss model with sample mean and sample standard deviation."""

    model_name = "gaussian"
    minimum_sample_size = 2

    def _calculate(
        self, confidence_level: float, losses: np.ndarray
    ) -> tuple[float, float, dict[str, float | int | str | None]]:
        location = float(np.mean(losses))
        scale = float(np.std(losses, ddof=1))
        z_score = float(norm.ppf(confidence_level))
        value_at_risk = location + scale * z_score
        expected_shortfall = location + scale * float(norm.pdf(z_score)) / (
            1.0 - confidence_level
        )
        return (
            value_at_risk,
            expected_shortfall,
            {
                "location_mean": location,
                "scale_sample_standard_deviation": scale,
                "standard_deviation_ddof": 1,
            },
        )
