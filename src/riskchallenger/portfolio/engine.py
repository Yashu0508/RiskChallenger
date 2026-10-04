"""Vectorized portfolio return, value, P&L, and loss calculations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from riskchallenger.data.loader import repository_root
from riskchallenger.logging import get_logger
from riskchallenger.portfolio.portfolio import Portfolio
from riskchallenger.portfolio.statistics import (
    PortfolioStatistics,
    calculate_statistics,
)

LOGGER = get_logger("portfolio.engine")
DEFAULT_RETURNS_PATH = Path("data/processed/nse50_returns.csv")


@dataclass(frozen=True)
class PortfolioCalculationResult:
    """Aligned time series produced by a portfolio calculation."""

    portfolio_returns: pd.Series
    portfolio_value: pd.Series
    pnl: pd.Series
    loss: pd.Series
    statistics: PortfolioStatistics

    def to_frame(self) -> pd.DataFrame:
        """Return all calculated daily series in one date-indexed table."""
        return pd.concat(
            [
                self.portfolio_returns.rename("portfolio_return"),
                self.portfolio_value.rename("portfolio_value"),
                self.pnl.rename("pnl"),
                self.loss.rename("loss"),
            ],
            axis=1,
        )


def load_return_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load Phase 1 decimal returns, indexed by parsed date, without imputation."""
    source_path = (
        Path(path) if path is not None else repository_root() / DEFAULT_RETURNS_PATH
    )
    if not source_path.is_absolute():
        source_path = repository_root() / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Portfolio return dataset not found: {source_path}")
    try:
        frame = pd.read_csv(source_path)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError(
            f"Could not parse portfolio return dataset: {source_path}"
        ) from error
    if frame.empty or "date" not in frame.columns:
        raise ValueError("Return dataset must contain rows and a 'date' column")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Return dataset contains duplicate columns: {duplicates}")
    try:
        dates = pd.to_datetime(frame.pop("date"), errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Return dataset contains invalid dates") from error
    if dates.isna().any():
        raise ValueError("Return dataset contains missing dates")
    frame.index = pd.DatetimeIndex(dates, name="date")
    if frame.index.has_duplicates:
        raise ValueError("Return dataset contains duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Return dataset dates must be in chronological order")
    LOGGER.info(
        "Loaded Phase 1 return dataset with %d rows and %d assets",
        len(frame),
        len(frame.columns),
    )
    return frame


def calculate_portfolio(
    portfolio: Portfolio, asset_returns: pd.DataFrame
) -> PortfolioCalculationResult:
    """Calculate portfolio returns and beginning-of-day-value P&L.

    A fully blank first observation across all selected assets is recognized as
    Phase 1's undefined initial return and excluded. Any partial or subsequent
    missing selected return fails; no financial return is imputed.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame")
    if asset_returns.empty:
        raise ValueError("asset return dataset is empty")
    if asset_returns.columns.duplicated().any():
        duplicates = asset_returns.columns[asset_returns.columns.duplicated()].tolist()
        raise ValueError(f"Return dataset contains duplicate assets: {duplicates}")
    if asset_returns.index.has_duplicates:
        raise ValueError("Return dataset contains duplicate dates")
    if not asset_returns.index.is_monotonic_increasing:
        raise ValueError("Return dataset index must be in chronological order")

    missing_assets = [
        asset for asset in portfolio.assets if asset not in asset_returns.columns
    ]
    if missing_assets:
        LOGGER.error("Requested assets are missing from returns: %s", missing_assets)
        raise ValueError(
            f"Requested assets not found in return dataset: {missing_assets}"
        )

    selected = asset_returns.loc[:, list(portfolio.assets)].copy()
    try:
        selected = selected.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Selected asset returns contain non-numeric values") from error

    if selected.empty:
        raise ValueError("No return observations available for selected assets")
    if selected.iloc[0].isna().all():
        LOGGER.info("Dropping the undefined first return observation from Phase 1")
        selected = selected.iloc[1:]
    elif selected.iloc[0].isna().any():
        missing = selected.columns[selected.iloc[0].isna()].tolist()
        raise ValueError(
            "Initial return row must be missing for all selected assets or none; "
            f"missing for {missing}"
        )

    missing_mask = selected.isna()
    if missing_mask.any().any():
        missing_details = {
            str(date): selected.columns[missing_mask.loc[date]].tolist()
            for date in selected.index[missing_mask.any(axis=1)]
        }
        raise ValueError(f"Missing selected asset returns detected: {missing_details}")
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        locations = np.argwhere(~np.isfinite(values))
        details = [
            (str(selected.index[row]), selected.columns[column])
            for row, column in locations[:10]
        ]
        raise ValueError(f"Non-finite selected asset returns detected: {details}")
    if selected.empty:
        raise ValueError(
            "No defined portfolio returns remain after initial-row handling"
        )

    LOGGER.info("Calculating portfolio returns for assets %s", portfolio.assets)
    portfolio_return_values = values @ np.asarray(portfolio.weight_vector, dtype=float)
    if not np.isfinite(portfolio_return_values).all():
        raise ValueError("Portfolio return calculation produced non-finite values")
    index = selected.index.copy()
    portfolio_returns = pd.Series(
        portfolio_return_values, index=index, name="portfolio_return"
    )

    portfolio_value = portfolio.initial_value * (1.0 + portfolio_returns).cumprod()
    prior_value = portfolio_value.shift(1)
    prior_value.iloc[0] = portfolio.initial_value
    pnl = prior_value * portfolio_returns
    loss = -pnl
    for series_name, series in (
        ("portfolio value", portfolio_value),
        ("P&L", pnl),
        ("loss", loss),
    ):
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError(f"{series_name} calculation produced non-finite values")

    statistics = calculate_statistics(
        portfolio_returns, portfolio_value, pnl, portfolio.initial_value
    )
    LOGGER.info("Calculated %d portfolio observations", len(portfolio_returns))
    LOGGER.info("Completed portfolio P&L and value calculations")
    return PortfolioCalculationResult(
        portfolio_returns=portfolio_returns,
        portfolio_value=portfolio_value.rename("portfolio_value"),
        pnl=pnl.rename("pnl"),
        loss=loss.rename("loss"),
        statistics=statistics,
    )
