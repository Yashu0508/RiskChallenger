"""Historical empirical VaR and fractional-tail Expected Shortfall."""

import numpy as np

from riskchallenger.risk.base import RiskModel


class HistoricalRiskModel(RiskModel):
    """Empirical model using a linear quantile and an integrated upper tail."""

    model_name = "historical"
    minimum_sample_size = 1

    def _calculate(
        self, confidence_level: float, losses: np.ndarray
    ) -> tuple[float, float, dict[str, float | int | str | None]]:
        value_at_risk = float(np.quantile(losses, confidence_level, method="linear"))
        sorted_losses = np.sort(losses)[::-1]
        tail_mass = float(losses.size * (1.0 - confidence_level))
        full_observations = int(np.floor(tail_mass))
        fractional_observation = tail_mass - full_observations
        weighted_tail_sum = float(sorted_losses[:full_observations].sum())
        if fractional_observation > 0.0:
            weighted_tail_sum += fractional_observation * float(
                sorted_losses[full_observations]
            )
        expected_shortfall = weighted_tail_sum / tail_mass
        return (
            value_at_risk,
            expected_shortfall,
            {
                "quantile_method": "numpy linear interpolation (method='linear')",
                "expected_shortfall_method": (
                    "integrated empirical upper tail with fractional "
                    "boundary observation"
                ),
                "tail_observation_mass": tail_mass,
            },
        )
