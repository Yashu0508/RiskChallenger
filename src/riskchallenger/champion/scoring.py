"""Configurable component weights and model scoring helpers."""

from dataclasses import dataclass

from riskchallenger.champion.validation import validate_weights


@dataclass(frozen=True)
class ScoreWeights:
    """Default composite-score weights, explicitly configurable by callers."""

    coverage: float = 0.40
    exception_rate: float = 0.25
    stability: float = 0.20
    operational_cost: float = 0.15

    def __post_init__(self) -> None:
        values = validate_weights(
            (self.coverage, self.exception_rate, self.stability, self.operational_cost)
        )
        for name, value in zip(
            ("coverage", "exception_rate", "stability", "operational_cost"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)


def exception_rate_score(observed_rate: float, expected_rate: float) -> float:
    """Normalize absolute exception-rate error to [0, 1].

    An error equal to the target rate or larger receives zero; exact calibration
    receives one. The normalization depends on confidence, not model identity.
    """
    return max(0.0, 1.0 - abs(observed_rate - expected_rate) / expected_rate)


def operational_score(runtime_seconds: float, time_scale_seconds: float) -> float:
    """Map runtime to [0, 1] with a documented hyperbolic cost scale."""
    return 1.0 / (1.0 + runtime_seconds / time_scale_seconds)
