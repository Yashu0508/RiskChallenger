"""Validated portfolio definition and explicit weight conventions."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from riskchallenger.logging import get_logger

LOGGER = get_logger("portfolio.definition")


@dataclass(frozen=True)
class Portfolio:
    """Portfolio assets, matching weights, and starting value.

    Weights are never normalized. By default the portfolio is long-only and
    fully invested. Set either validation option explicitly for another policy.
    """

    assets: Sequence[str]
    weights: Mapping[str, object]
    initial_value: float
    long_only: bool = True
    fully_invested: bool = True
    weight_tolerance: float = 1e-8
    _weight_vector: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.assets, (str, bytes)):
            raise ValueError("assets must be a sequence of unique asset names")
        assets = tuple(self.assets)
        if not assets:
            raise ValueError("at least one asset is required")
        if any(not isinstance(asset, str) or not asset.strip() for asset in assets):
            raise ValueError("asset names must be non-empty strings")
        if len(assets) != len(set(assets)):
            raise ValueError("duplicate asset names are not allowed")
        object.__setattr__(self, "assets", assets)

        if not isinstance(self.weights, Mapping):
            raise ValueError("weights must be a mapping from asset name to weight")
        missing = [asset for asset in assets if asset not in self.weights]
        unexpected = [asset for asset in self.weights if asset not in assets]
        if missing:
            raise ValueError(f"missing weights for assets: {missing}")
        if unexpected:
            raise ValueError(f"weights provided for unselected assets: {unexpected}")

        try:
            tolerance = float(self.weight_tolerance)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "weight_tolerance must be a finite non-negative number"
            ) from error
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("weight_tolerance must be a finite non-negative number")
        object.__setattr__(self, "weight_tolerance", tolerance)

        normalized_weights: dict[str, float] = {}
        for asset in assets:
            raw_weight = self.weights[asset]
            if isinstance(raw_weight, bool):
                raise ValueError(f"weight for {asset} must be numeric, not boolean")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as error:
                raise ValueError(f"weight for {asset} must be numeric") from error
            if not math.isfinite(weight):
                raise ValueError(f"weight for {asset} must be finite")
            if self.long_only and weight < 0:
                raise ValueError(
                    f"weight for {asset} must be non-negative in long-only mode"
                )
            normalized_weights[asset] = weight

        weight_sum = math.fsum(normalized_weights.values())
        if self.fully_invested and not math.isclose(
            weight_sum, 1.0, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(
                f"weights must sum to 1.0 in fully invested mode; got {weight_sum:.16g}"
            )

        object.__setattr__(self, "weights", MappingProxyType(normalized_weights))
        object.__setattr__(
            self, "_weight_vector", tuple(normalized_weights[asset] for asset in assets)
        )
        try:
            initial_value = float(self.initial_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "initial_value must be a finite positive number"
            ) from error
        if not math.isfinite(initial_value) or initial_value <= 0:
            raise ValueError("initial_value must be a finite positive number")
        object.__setattr__(self, "initial_value", initial_value)
        LOGGER.info("Created portfolio with assets %s", ", ".join(assets))
        LOGGER.info(
            "Validated portfolio weights (long_only=%s, fully_invested=%s)",
            self.long_only,
            self.fully_invested,
        )

    @property
    def weight_vector(self) -> tuple[float, ...]:
        """Weights ordered exactly as ``assets``."""
        return self._weight_vector
