"""Interpretable, causal features and one-step-ahead variance labels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "lag_return_1",
    "lag_return_2",
    "lag_return_5",
    "lag_return_10",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "rolling_mean_5d",
    "rolling_mean_20d",
    "downside_volatility_20d",
    "maximum_loss_20d",
    "maximum_drawdown_20d",
    "cumulative_return_5d",
)
TARGET_COLUMN = "next_return_squared"


def _maximum_drawdown(window: np.ndarray) -> float:
    levels = np.cumprod(1.0 + window)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], levels)))[1:]
    return float(np.min(levels / peaks - 1.0))


@dataclass(frozen=True)
class RiskFeatureBuilder:
    """Build features known at time t and the t+1 squared-return target.

    Lag 1 is the return observed at the forecast origin. Rolling features use
    observations through that origin, never later observations.
    """

    volatility_windows: tuple[int, ...] = (5, 10, 20)
    mean_windows: tuple[int, ...] = (5, 20)
    target_column: str = TARGET_COLUMN

    def __post_init__(self) -> None:
        if self.volatility_windows != (5, 10, 20):
            raise ValueError(
                "the initial feature set uses volatility windows (5, 10, 20)"
            )
        if self.mean_windows != (5, 20):
            raise ValueError("the initial feature set uses mean windows (5, 20)")
        if self.target_column != TARGET_COLUMN:
            raise ValueError(f"target_column must be {TARGET_COLUMN!r}")

    @property
    def maximum_lookback(self) -> int:
        return 20

    @staticmethod
    def validate_returns(returns: pd.Series) -> pd.Series:
        if not isinstance(returns, pd.Series):
            raise TypeError("portfolio returns must be a pandas Series")
        if returns.empty:
            raise ValueError("portfolio returns must not be empty")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("portfolio return index must be a DatetimeIndex")
        if returns.index.hasnans or returns.index.has_duplicates:
            raise ValueError("portfolio return dates must be valid and unique")
        if not returns.index.is_monotonic_increasing:
            raise ValueError("portfolio returns must be in chronological order")
        if returns.index.tz is not None:
            raise ValueError("portfolio return dates must be timezone-naive")
        try:
            values = pd.to_numeric(returns, errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("portfolio returns must be numeric") from error
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("portfolio returns must be finite; no imputation is done")
        if (values.to_numpy() <= -1.0).any():
            raise ValueError("portfolio simple returns must be greater than -1")
        values.name = returns.name or "portfolio_return"
        return values

    def build_features(
        self, returns: pd.Series, *, include_target: bool = True
    ) -> pd.DataFrame:
        """Build date-indexed features; retain warm-up/terminal NaNs explicitly."""
        series = self.validate_returns(returns)
        frame = pd.DataFrame(index=series.index)
        frame["lag_return_1"] = series
        frame["lag_return_2"] = series.shift(1)
        frame["lag_return_5"] = series.shift(4)
        frame["lag_return_10"] = series.shift(9)
        for window in self.volatility_windows:
            frame[f"volatility_{window}d"] = series.rolling(window).std(ddof=1)
        for window in self.mean_windows:
            frame[f"rolling_mean_{window}d"] = series.rolling(window).mean()
        negative_squared = series.clip(upper=0.0).pow(2)
        frame["downside_volatility_20d"] = np.sqrt(negative_squared.rolling(20).mean())
        frame["maximum_loss_20d"] = (-series).clip(lower=0.0).rolling(20).max()
        frame["maximum_drawdown_20d"] = series.rolling(20).apply(
            _maximum_drawdown, raw=True
        )
        frame["cumulative_return_5d"] = (1.0 + series).rolling(5).apply(
            np.prod, raw=True
        ) - 1.0
        if include_target:
            frame[self.target_column] = series.shift(-1).pow(2)
        return frame

    def build_latest_features(self, returns: pd.Series) -> pd.DataFrame:
        """Efficiently calculate only the latest causal feature row."""
        series = self.validate_returns(returns)
        if len(series) < self.maximum_lookback:
            raise ValueError(
                "at least 20 returns are required to build current features"
            )
        latest = self.build_features(
            series.iloc[-self.maximum_lookback :], include_target=False
        )
        return latest.loc[:, list(FEATURE_COLUMNS)].tail(1)
