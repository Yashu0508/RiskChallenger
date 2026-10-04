"""Polymorphic scenario definitions for the stress-testing engine."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import numpy as np
import pandas as pd

from riskchallenger.stress.validation import (
    align_shocks,
    normalize_shock_pairs,
    parse_scenario_date,
    validate_name,
    validate_return_shock,
)


@dataclass(frozen=True)
class ResolvedScenario:
    """Validated per-asset returns and audit metadata for one scenario."""

    name: str
    scenario_type: str
    asset_returns: tuple[tuple[str, float], ...]
    assumptions: tuple[str, ...]
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None


class StressScenario(Protocol):
    """Scenario interface; implementations resolve shocks for portfolio assets."""

    name: str
    scenario_type: str

    def resolve(self, assets: Sequence[str]) -> ResolvedScenario: ...


@dataclass(frozen=True)
class UniformShock:
    """Apply one simple return shock to every portfolio asset."""

    return_shock: float
    name: str = "uniform_market_shock"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_name(self.name))
        object.__setattr__(
            self, "return_shock", validate_return_shock(self.return_shock)
        )

    @property
    def scenario_type(self) -> str:
        return "uniform_market_shock"

    def resolve(self, assets: Sequence[str]) -> ResolvedScenario:
        return ResolvedScenario(
            name=self.name,
            scenario_type=self.scenario_type,
            asset_returns=tuple((asset, self.return_shock) for asset in assets),
            assumptions=(
                f"Every selected asset receives the same simple return "
                f"shock ({self.return_shock:.12g}).",
                "Portfolio weights are held fixed for this one-period mark-to-market.",
            ),
        )


@dataclass(frozen=True)
class AssetSpecificShock:
    """Apply explicitly specified simple return shocks to each asset."""

    asset_shocks: Mapping[str, object] | Sequence[tuple[str, object]]
    name: str = "asset_specific_shock"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_name(self.name))
        object.__setattr__(
            self,
            "asset_shocks",
            normalize_shock_pairs(self.asset_shocks, "asset_shocks"),
        )

    @property
    def scenario_type(self) -> str:
        return "asset_specific_shock"

    def resolve(self, assets: Sequence[str]) -> ResolvedScenario:
        aligned = align_shocks(assets, self.asset_shocks, "asset-specific scenario")
        return ResolvedScenario(
            name=self.name,
            scenario_type=self.scenario_type,
            asset_returns=tuple(aligned.items()),
            assumptions=(
                "Each selected asset receives its explicitly specified simple return.",
                "Portfolio weights are held fixed for this one-period mark-to-market.",
            ),
        )


@dataclass(frozen=True)
class HistoricalReplayScenario:
    """Replay a supplied vector or cumulative historical price-window return."""

    asset_returns: Mapping[str, object] | Sequence[tuple[str, object]]
    name: str = "historical_replay"
    start_date: date | datetime | str | pd.Timestamp | None = None
    end_date: date | datetime | str | pd.Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_name(self.name))
        object.__setattr__(
            self,
            "asset_returns",
            normalize_shock_pairs(self.asset_returns, "historical_returns"),
        )
        start = (
            parse_scenario_date(self.start_date, "start_date")
            if self.start_date is not None
            else None
        )
        end = (
            parse_scenario_date(self.end_date, "end_date")
            if self.end_date is not None
            else None
        )
        if (start is None) != (end is None):
            raise ValueError("historical replay requires both start_date and end_date")
        if start is not None and start >= end:
            raise ValueError("historical replay start_date must precede end_date")
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)

    @property
    def scenario_type(self) -> str:
        return "historical_replay"

    @classmethod
    def from_price_window(
        cls,
        prices: pd.DataFrame,
        name: str,
        start_date: date | datetime | str | pd.Timestamp,
        end_date: date | datetime | str | pd.Timestamp,
        assets: Sequence[str] | None = None,
    ) -> "HistoricalReplayScenario":
        """Derive cumulative event returns from canonical price observations.

        Uses the last close strictly before ``start_date`` as the base and the
        last close on or before ``end_date`` as the event endpoint.
        """
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            raise ValueError("prices must be a non-empty DataFrame")
        if prices.columns.duplicated().any():
            raise ValueError("price data contain duplicate asset columns")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("price data must have a DatetimeIndex")
        if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
            raise ValueError("price dates must be unique and chronological")
        if prices.index.tz is not None:
            raise ValueError("price dates must be timezone-naive")
        start = parse_scenario_date(start_date, "start_date")
        end = parse_scenario_date(end_date, "end_date")
        if start >= end:
            raise ValueError("historical replay start_date must precede end_date")
        selected_assets = tuple(assets) if assets is not None else tuple(prices.columns)
        if not selected_assets:
            raise ValueError("historical replay assets must not be empty")
        if len(set(selected_assets)) != len(selected_assets):
            raise ValueError("historical replay asset list contains duplicates")
        missing_assets = [asset for asset in selected_assets if asset not in prices]
        if missing_assets:
            raise ValueError(f"price data missing assets: {missing_assets}")
        before = prices.index < start
        through_end = prices.index <= end
        if not before.any():
            raise ValueError("price data contain no close before historical start_date")
        if not through_end.any():
            raise ValueError(
                "price data contain no close on or before historical end_date"
            )
        base_date = prices.index[before][-1]
        event_end = prices.index[through_end][-1]
        if event_end < start:
            raise ValueError(
                "price data contain no observation inside the event window"
            )
        try:
            base = pd.to_numeric(
                prices.loc[base_date, list(selected_assets)], errors="raise"
            )
            final = pd.to_numeric(
                prices.loc[event_end, list(selected_assets)], errors="raise"
            )
        except (TypeError, ValueError) as error:
            raise ValueError("historical endpoint prices must be numeric") from error
        base_values = base.to_numpy(dtype=float)
        final_values = final.to_numpy(dtype=float)
        if not np.isfinite(base_values).all() or not np.isfinite(final_values).all():
            raise ValueError(
                "historical endpoint prices contain missing/non-finite data"
            )
        if (base_values <= 0).any() or (final_values <= 0).any():
            raise ValueError("historical endpoint prices must be positive")
        returns = final_values / base_values - 1.0
        return cls(
            asset_returns=tuple(zip(selected_assets, returns, strict=True)),
            name=name,
            start_date=start,
            end_date=end,
        )

    def resolve(self, assets: Sequence[str]) -> ResolvedScenario:
        aligned = align_shocks(assets, self.asset_returns, "historical replay")
        return ResolvedScenario(
            name=self.name,
            scenario_type=self.scenario_type,
            asset_returns=tuple(aligned.items()),
            assumptions=(
                "Today's portfolio weights are applied to the specified historical "
                "asset-return vector; historical portfolio weights are not reused.",
                "Returns are simple cumulative endpoint returns for the event window.",
                "For price-window construction, base is the last close before the "
                "start date; endpoint is the last close on or before the end date.",
            ),
            start_date=self.start_date,
            end_date=self.end_date,
        )


@dataclass(frozen=True)
class VolatilityShock:
    """Scale an explicit reference return realization by a volatility factor."""

    reference_returns: Mapping[str, object] | Sequence[tuple[str, object]]
    volatility_multiplier: float
    name: str = "volatility_shock"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_name(self.name))
        object.__setattr__(
            self,
            "reference_returns",
            normalize_shock_pairs(self.reference_returns, "reference_returns"),
        )
        if isinstance(self.volatility_multiplier, bool):
            raise ValueError("volatility_multiplier must be finite and >= 1")
        try:
            multiplier = float(self.volatility_multiplier)
        except (TypeError, ValueError) as error:
            raise ValueError("volatility_multiplier must be finite and >= 1") from error
        if not np.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError("volatility_multiplier must be finite and >= 1")
        object.__setattr__(self, "volatility_multiplier", multiplier)

    @property
    def scenario_type(self) -> str:
        return "volatility_scaled_realization"

    def resolve(self, assets: Sequence[str]) -> ResolvedScenario:
        reference = align_shocks(assets, self.reference_returns, "reference returns")
        scaled = {
            asset: validate_return_shock(
                shock * self.volatility_multiplier,
                f"volatility-scaled return[{asset}]",
            )
            for asset, shock in reference.items()
        }
        return ResolvedScenario(
            name=self.name,
            scenario_type=self.scenario_type,
            asset_returns=tuple(scaled.items()),
            assumptions=(
                "Volatility shock multiplies each supplied reference return by "
                f"{self.volatility_multiplier:.12g}, centered at zero.",
                "This is a deterministic scaled realization, not a probabilistic "
                "volatility forecast or unique price path.",
            ),
        )


@dataclass(frozen=True)
class JointShock:
    """Deterministic common market shock plus per-asset idiosyncratic shocks."""

    market_shock: float
    idiosyncratic_shocks: Mapping[str, object] | Sequence[tuple[str, object]] = ()
    name: str = "joint_market_idiosyncratic_shock"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_name(self.name))
        object.__setattr__(
            self,
            "market_shock",
            validate_return_shock(self.market_shock, "market_shock"),
        )
        shocks = self.idiosyncratic_shocks
        if isinstance(shocks, Mapping):
            pairs = list(shocks.items())
        else:
            try:
                pairs = list(shocks)
            except TypeError as error:
                raise ValueError(
                    "idiosyncratic_shocks must be mapping or pairs"
                ) from error
        normalized: list[tuple[str, float]] = []
        if pairs:
            normalized.extend(normalize_shock_pairs(pairs, "idiosyncratic_shocks"))
        object.__setattr__(self, "idiosyncratic_shocks", tuple(normalized))

    @property
    def scenario_type(self) -> str:
        return "deterministic_joint_shock"

    def resolve(self, assets: Sequence[str]) -> ResolvedScenario:
        unknown = [
            asset for asset, _ in self.idiosyncratic_shocks if asset not in assets
        ]
        if unknown:
            raise ValueError(f"joint shock references unknown assets: {unknown}")
        idiosyncratic = dict(self.idiosyncratic_shocks)
        combined = tuple(
            (
                asset,
                validate_return_shock(
                    self.market_shock + idiosyncratic.get(asset, 0.0),
                    f"combined shock[{asset}]",
                ),
            )
            for asset in assets
        )
        return ResolvedScenario(
            name=self.name,
            scenario_type=self.scenario_type,
            asset_returns=combined,
            assumptions=(
                "Each asset shock is the common market shock plus its specified "
                "idiosyncratic return shock.",
                "Unspecified idiosyncratic shocks are zero; common component creates "
                "deterministic joint co-movement.",
                "This is a scenario construction, not a stochastic correlation model.",
            ),
        )
