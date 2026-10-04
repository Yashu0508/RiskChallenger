"""Mathematically derived simple daily returns."""

import math


def calculate_returns(
    rows: list[dict[str, str | None]], assets: list[str], date_column: str = "date"
) -> list[dict[str, str | float | None]]:
    """Calculate simple returns in decimal units, leaving each initial return blank."""
    output: list[dict[str, str | float | None]] = []
    previous: dict[str, float] | None = None
    for row in rows:
        current_row: dict[str, str | float | None] = {date_column: row[date_column]}
        current: dict[str, float] = {}
        for asset in assets:
            raw = row.get(asset)
            if raw is None or not raw.strip():
                raise ValueError(
                    "Cannot calculate returns: missing price for "
                    f"{asset} on {row[date_column]}"
                )
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Cannot calculate returns: invalid price for "
                    f"{asset} on {row[date_column]}"
                )
            current[asset] = value
            if previous is None:
                current_row[asset] = None
            else:
                daily_return = value / previous[asset] - 1.0
                if not math.isfinite(daily_return):
                    raise ValueError(
                        f"Non-finite return for {asset} on {row[date_column]}"
                    )
                current_row[asset] = daily_return
        output.append(current_row)
        previous = current
    return output


def validate_returns(
    prices: list[dict[str, str | None]],
    returns: list[dict[str, str | float | None]],
    assets: list[str],
    date_column: str = "date",
) -> dict[str, object]:
    """Check return dimensions, initial undefined values, and price consistency."""
    errors: list[str] = []
    if len(prices) != len(returns):
        errors.append("Return row count does not match price row count")
    for index, (price_row, return_row) in enumerate(zip(prices, returns)):
        if price_row.get(date_column) != return_row.get(date_column):
            errors.append(f"Date mismatch at row {index}")
        for asset in assets:
            observed = return_row.get(asset)
            if index == 0:
                if observed is not None:
                    errors.append(f"Initial return for {asset} must be undefined")
                continue
            try:
                expected = (
                    float(price_row[asset]) / float(prices[index - 1][asset]) - 1.0
                )
                value = float(observed)
            except (TypeError, ValueError, KeyError, ZeroDivisionError):
                errors.append(
                    f"Return unavailable or invalid for {asset} at row {index}"
                )
                continue
            if not math.isfinite(value):
                errors.append(f"Non-finite return for {asset} at row {index}")
            elif not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
                errors.append(
                    f"Return does not match price formula for {asset} at row {index}"
                )
    return {
        "status": "passed" if not errors else "failed",
        "row_count": len(returns),
        "asset_count": len(assets),
        "initial_undefined_observations": len(assets) if prices else 0,
        "units": "decimal simple returns; 0.01 represents 1%",
        "formula": "P_t / P_(t-1) - 1",
        "errors": errors,
    }
