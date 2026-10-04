"""Empirical-bootstrap Monte Carlo VaR and Expected Shortfall."""

from numbers import Integral

import numpy as np

from riskchallenger.risk.base import RiskModel
from riskchallenger.risk.historical import HistoricalRiskModel


class MonteCarloRiskModel(RiskModel):
    """Bootstrap portfolio losses with replacement using a seeded Generator.

    Each simulated one-period loss is sampled from the supplied historical loss
    observations. VaR and ES are then delegated to the historical model so both
    use the same linear quantile and fractional empirical tail conventions.
    """

    model_name = "monte_carlo"
    minimum_sample_size = 20

    def __init__(self, n_simulations: int = 100_000, random_seed: int = 2026) -> None:
        super().__init__()
        if isinstance(n_simulations, bool) or not isinstance(n_simulations, Integral):
            raise ValueError("n_simulations must be a positive integer")
        if n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer")
        if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
            raise ValueError("random_seed must be a non-negative integer")
        if random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        self.n_simulations = int(n_simulations)
        self.random_seed = int(random_seed)

    def _calculate(
        self, confidence_level: float, losses: np.ndarray
    ) -> tuple[float, float, dict[str, float | int | str | None]]:
        generator = np.random.default_rng(self.random_seed)
        simulated_losses = generator.choice(
            losses, size=self.n_simulations, replace=True
        )
        if self._unit is None:
            raise RuntimeError("Monte Carlo model has not been fitted")
        empirical_result = (
            HistoricalRiskModel()
            .fit(simulated_losses, unit=self._unit)
            .calculate(confidence_level)
        )
        return (
            empirical_result.var,
            empirical_result.expected_shortfall,
            {
                "methodology": "empirical bootstrap with replacement",
                "n_simulations": self.n_simulations,
                "random_seed": self.random_seed,
                "historical_sample_size": int(losses.size),
                "quantile_method": empirical_result.model_parameters["quantile_method"],
                "expected_shortfall_method": empirical_result.model_parameters[
                    "expected_shortfall_method"
                ],
            },
        )

    def _result_sample_size(self) -> int:
        """Report the number of simulated losses used for the tail estimate."""
        return self.n_simulations
