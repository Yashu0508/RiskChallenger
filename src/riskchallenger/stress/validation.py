"""Validation helpers for explicit stress scenario assumptions."""

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime

import pandas as pd


def validate_name(name: str) -> str:
    """Require a human-readable, nonempty scenario name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("scenario name must be a non-empty string")
    return name.strip()


def validate_return_shock(value: object, name: str = "return shock") -> float:
    """Require a finite simple return that preserves a positive price."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= -1.0:
        raise ValueError(f"{name} must be greater than -1 to keep price positive")
    return result


def normalize_shock_pairs(
    shocks: Mapping[str, object] | Sequence[tuple[str, object]], name: str
) -> tuple[tuple[str, float], ...]:
    """Normalize scenario shocks while preserving duplicate detection."""
    if isinstance(shocks, Mapping):
        pairs = list(shocks.items())
    else:
        try:
            pairs = list(shocks)
        except TypeError as error:
            raise ValueError(
                f"{name} must be an asset-to-return mapping or pairs"
            ) from error
    if not pairs:
        raise ValueError(f"{name} must not be empty")
    normalized: list[tuple[str, float]] = []
    seen: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{name} entries must be (asset, return) pairs")
        asset, shock = pair
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"{name} asset names must be non-empty strings")
        asset = asset.strip()
        if asset in seen:
            raise ValueError(f"duplicate asset in {name}: {asset}")
        seen.add(asset)
        normalized.append((asset, validate_return_shock(shock, f"{name}[{asset}]")))
    return tuple(normalized)


def align_shocks(
    portfolio_assets: Sequence[str], shocks: Sequence[tuple[str, float]], name: str
) -> dict[str, float]:
    """Require one shock for every selected asset, with no unknown assets."""
    supplied = dict(shocks)
    unknown = [asset for asset in supplied if asset not in portfolio_assets]
    missing = [asset for asset in portfolio_assets if asset not in supplied]
    if unknown:
        raise ValueError(f"{name} references unknown portfolio assets: {unknown}")
    if missing:
        raise ValueError(f"{name} is missing shocks for portfolio assets: {missing}")
    return {asset: supplied[asset] for asset in portfolio_assets}


def parse_scenario_date(value: date | datetime | str | pd.Timestamp, name: str):
    """Parse a timezone-naive date boundary."""
    try:
        result = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid date") from error
    if pd.isna(result) or result.tz is not None:
        raise ValueError(f"{name} must be a valid timezone-naive date")
    return result
