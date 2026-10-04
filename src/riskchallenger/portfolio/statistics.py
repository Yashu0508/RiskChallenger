"""Basic descriptive statistics for portfolio return and value series."""

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class PortfolioStatistics:
    """Descriptive portfolio results; return standard deviation is sample-based."""

    mean_daily_return: float
    standard_deviation_daily_return: float
    cumulative_return: float
    minimum_daily_return: float
    maximum_daily_return: float
    initial_portfolio_value: float
    final_portfolio_value: float
    total_pnl: float

    def to_dict(self) -> dict[str, float]:
        """Return statistics in a JSON-friendly scalar mapping."""
        return asdict(self)


def calculate_statistics(
    portfolio_returns: pd.Series,
    portfolio_value: pd.Series,
    pnl: pd.Series,
    initial_value: float,
) -> PortfolioStatistics:
    """Calculate arithmetic daily moments and compounded total performance."""
    if portfolio_returns.empty:
        raise ValueError("cannot calculate portfolio statistics for an empty series")
    final_value = float(portfolio_value.iloc[-1])
    return PortfolioStatistics(
        mean_daily_return=float(portfolio_returns.mean()),
        standard_deviation_daily_return=float(portfolio_returns.std(ddof=1)),
        cumulative_return=final_value / initial_value - 1.0,
        minimum_daily_return=float(portfolio_returns.min()),
        maximum_daily_return=float(portfolio_returns.max()),
        initial_portfolio_value=float(initial_value),
        final_portfolio_value=final_value,
        total_pnl=float(pnl.sum()),
    )
