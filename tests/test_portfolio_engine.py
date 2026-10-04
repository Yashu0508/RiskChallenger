"""Tests for portfolio definition, returns, P&L, and value evolution."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskchallenger.portfolio import Portfolio, calculate_portfolio, load_return_dataset


def make_returns(values: dict[str, list[float | None]]) -> pd.DataFrame:
    length = len(next(iter(values.values())))
    index = pd.date_range("2024-01-01", periods=length, name="date")
    return pd.DataFrame(values, index=index)


def test_valid_long_only_fully_invested_weights() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 0.6, "B": 0.4}, 1_000_000)
    assert portfolio.assets == ("A", "B")
    assert portfolio.weight_vector == (0.6, 0.4)
    assert portfolio.initial_value == 1_000_000


@pytest.mark.parametrize(
    "weights", [{"A": 0.4, "B": 0.4, "C": 0.1}, {"A": 0.6, "B": 0.3}]
)
def test_weights_must_sum_to_one_without_normalization(
    weights: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        Portfolio(list(weights), weights, 100)


def test_negative_weight_rejected_in_long_only_mode() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Portfolio(["A", "B"], {"A": 1.1, "B": -0.1}, 100)


def test_short_weight_allowed_when_long_only_disabled() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 1.1, "B": -0.1}, 100, long_only=False)
    assert portfolio.weight_vector == (1.1, -0.1)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_weights_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Portfolio(["A"], {"A": weight}, 100)


def test_non_numeric_weight_rejected() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        Portfolio(["A"], {"A": "not-a-number"}, 100)


def test_missing_and_extra_weights_rejected() -> None:
    with pytest.raises(ValueError, match="missing weights"):
        Portfolio(["A", "B"], {"A": 1.0}, 100)
    with pytest.raises(ValueError, match="unselected assets"):
        Portfolio(["A"], {"A": 0.5, "B": 0.5}, 100)


def test_duplicate_assets_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate asset"):
        Portfolio(["A", "A"], {"A": 1.0}, 100)


@pytest.mark.parametrize("initial_value", [0, -1, float("nan"), float("inf")])
def test_initial_value_must_be_finite_and_positive(initial_value: float) -> None:
    with pytest.raises(ValueError, match="initial_value"):
        Portfolio(["A"], {"A": 1.0}, initial_value)


def test_asset_order_aligns_with_weights() -> None:
    portfolio = Portfolio(["B", "A"], {"A": 0.6, "B": 0.4}, 1_000)
    returns = make_returns({"A": [0.10], "B": [0.0]})
    result = calculate_portfolio(portfolio, returns)
    assert result.portfolio_returns.iloc[0] == pytest.approx(0.06)


def test_unknown_and_duplicate_assets_in_return_data_fail() -> None:
    portfolio = Portfolio(["A"], {"A": 1.0}, 100)
    with pytest.raises(ValueError, match="not found"):
        calculate_portfolio(portfolio, make_returns({"B": [0.01]}))
    duplicated = pd.DataFrame([[0.1, 0.2]], columns=["A", "A"])
    with pytest.raises(ValueError, match="duplicate assets"):
        calculate_portfolio(portfolio, duplicated)


def test_manual_portfolio_return_pnl_loss_and_value() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 0.6, "B": 0.4}, 1_000_000)
    asset_returns = make_returns({"A": [0.10, -0.10], "B": [0.0, -0.10]})
    result = calculate_portfolio(portfolio, asset_returns)

    assert result.portfolio_returns.tolist() == pytest.approx([0.06, -0.10])
    assert result.pnl.tolist() == pytest.approx([60_000, -106_000])
    assert result.loss.tolist() == pytest.approx([-60_000, 106_000])
    assert result.portfolio_value.tolist() == pytest.approx([1_060_000, 954_000])
    assert result.to_frame().columns.tolist() == [
        "portfolio_return",
        "portfolio_value",
        "pnl",
        "loss",
    ]


def test_initial_phase_one_undefined_row_is_dropped_without_imputation() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 0.5, "B": 0.5}, 100)
    returns = make_returns({"A": [None, 0.1], "B": [None, 0.0]})
    result = calculate_portfolio(portfolio, returns)
    assert len(result.portfolio_returns) == 1
    assert result.portfolio_returns.iloc[0] == pytest.approx(0.05)
    assert result.portfolio_returns.index[0] == returns.index[1]


def test_partial_or_later_missing_returns_fail_clearly() -> None:
    portfolio = Portfolio(["A", "B"], {"A": 0.5, "B": 0.5}, 100)
    partial_initial = make_returns({"A": [None, 0.1], "B": [0.0, 0.0]})
    with pytest.raises(ValueError, match="missing for all selected assets"):
        calculate_portfolio(portfolio, partial_initial)
    later_missing = make_returns({"A": [None, 0.1], "B": [None, np.nan]})
    with pytest.raises(ValueError, match="Missing selected asset returns"):
        calculate_portfolio(portfolio, later_missing)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_infinite_asset_returns_fail(bad_value: float) -> None:
    portfolio = Portfolio(["A"], {"A": 1.0}, 100)
    with pytest.raises(ValueError, match="Non-finite"):
        calculate_portfolio(portfolio, make_returns({"A": [bad_value]}))


def test_basic_statistics_and_sample_standard_deviation() -> None:
    portfolio = Portfolio(["A"], {"A": 1.0}, 100)
    result = calculate_portfolio(portfolio, make_returns({"A": [0.1, -0.1]}))
    stats = result.statistics
    assert stats.mean_daily_return == pytest.approx(0.0)
    assert stats.standard_deviation_daily_return == pytest.approx(
        np.std([0.1, -0.1], ddof=1)
    )
    assert stats.cumulative_return == pytest.approx(-0.01)
    assert stats.minimum_daily_return == pytest.approx(-0.1)
    assert stats.maximum_daily_return == pytest.approx(0.1)
    assert stats.initial_portfolio_value == 100
    assert stats.final_portfolio_value == pytest.approx(99)
    assert stats.total_pnl == pytest.approx(-1)


def test_load_actual_returns_dataset_and_run_engine() -> None:
    returns = load_return_dataset()
    assert returns.index.name == "date"
    assert returns.index.is_monotonic_increasing
    assert len(returns) == 4185

    selected_assets = ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"]
    portfolio = Portfolio(
        selected_assets,
        {asset: 1 / len(selected_assets) for asset in selected_assets},
        1_000_000,
    )
    result = calculate_portfolio(portfolio, returns)
    assert len(result.portfolio_returns) == 4184
    assert result.portfolio_returns.index.equals(returns.index[1:])
    assert np.isfinite(result.to_frame().to_numpy(dtype=float)).all()
    assert result.statistics.initial_portfolio_value == 1_000_000


def test_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Portfolio return dataset not found"):
        load_return_dataset(tmp_path / "missing.csv")
