"""Structural and financial-observation validation for wide price data."""

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation


@dataclass
class ValidationResult:
    """Machine-readable validation outcome and normalized structural rows."""

    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    information: list[str] = field(default_factory=list)
    schema: dict[str, object] = field(default_factory=dict)
    date_summary: dict[str, object] = field(default_factory=dict)
    duplicate_summary: dict[str, object] = field(default_factory=dict)
    missingness: dict[str, object] = field(default_factory=dict)
    invalid_price_summary: dict[str, object] = field(default_factory=dict)
    asset_metadata: list[dict[str, object]] = field(default_factory=list)
    rows: list[dict[str, str | None]] = field(default_factory=list, repr=False)
    assets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize the public report fields, excluding normalized data rows."""
        data = asdict(self)
        data.pop("rows")
        data.pop("assets")
        return data


def validate_prices(
    fieldnames: list[str], records: list[dict[str, str | None]]
) -> ValidationResult:
    """Validate the discovered date-plus-assets wide schema.

    Exact duplicate rows are structurally redundant and removed. Conflicting
    observations for a date are retained in diagnostics and make validation fail.
    Missing and invalid prices remain untouched and are reported.
    """
    result = ValidationResult(status="passed")
    if not fieldnames or fieldnames[0].strip().lower() != "date":
        result.errors.append("Expected the first column to be the date field")
        result.status = "failed"
        return result

    assets = fieldnames[1:]
    if not assets:
        result.errors.append("At least one asset price column is required")
    if len(assets) != len(set(assets)):
        result.errors.append("Asset columns must have unique names")
    result.assets = assets
    result.schema = {
        "format": "wide",
        "date_column": fieldnames[0],
        "asset_columns": assets,
        "column_count": len(fieldnames),
        "asset_count": len(assets),
        "source_value_types": (
            "CSV text; price lexemes validated as finite numeric values"
        ),
    }

    exact_row_keys: set[tuple[str | None, ...]] = set()
    exact_duplicate_count = 0
    parsed: list[tuple[date, dict[str, str | None]]] = []
    invalid_dates: list[dict[str, object]] = []
    date_rows: dict[date, list[dict[str, str | None]]] = {}
    original_valid_dates: list[date] = []

    for row_number, source_row in enumerate(records, start=2):
        row = {name: source_row.get(name) for name in fieldnames}
        key = tuple(row[name] for name in fieldnames)
        if key in exact_row_keys:
            exact_duplicate_count += 1
            continue
        exact_row_keys.add(key)

        raw_date = row.get(fieldnames[0])
        try:
            parsed_date = date.fromisoformat(raw_date.strip()) if raw_date else None
            if parsed_date is None:
                raise ValueError("empty date")
        except ValueError:
            invalid_dates.append({"row": row_number, "value": raw_date})
            continue
        parsed.append((parsed_date, row))
        original_valid_dates.append(parsed_date)
        date_rows.setdefault(parsed_date, []).append(row)

    duplicate_dates = [
        day.isoformat()
        for day, rows_at_date in date_rows.items()
        if len(rows_at_date) > 1
    ]
    if exact_duplicate_count:
        result.information.append(
            f"Removed {exact_duplicate_count} demonstrably exact duplicate row(s)."
        )
    if duplicate_dates:
        result.errors.append(
            "Conflicting non-identical observations exist for "
            f"{len(duplicate_dates)} date(s)."
        )
    if invalid_dates:
        result.errors.append(f"Found {len(invalid_dates)} invalid or missing date(s).")

    parsed.sort(key=lambda item: item[0])
    rows = [row for _, row in parsed]
    dates = [day for day, _ in parsed]
    out_of_order = any(
        a > b for a, b in zip(original_valid_dates, original_valid_dates[1:])
    )
    if out_of_order:
        result.information.append(
            "Rows were sorted chronologically as a structural correction."
        )

    calendar_gaps = [
        {
            "after": left.isoformat(),
            "before": right.isoformat(),
            "calendar_days": (right - left).days,
        }
        for left, right in zip(dates, dates[1:])
        if (right - left) > timedelta(days=3)
    ]
    result.date_summary = {
        "first_date": dates[0].isoformat() if dates else None,
        "last_date": dates[-1].isoformat() if dates else None,
        "valid_observation_dates": len(dates),
        "invalid_dates": invalid_dates,
        "duplicate_dates": duplicate_dates,
        "input_was_chronological": not out_of_order,
        "canonical_order": "ascending",
        "calendar_gaps_over_three_days": calendar_gaps,
        "calendar_gap_note": (
            "Calendar gaps are descriptive; without an exchange calendar they "
            "are not classified as missing trading sessions."
        ),
    }
    result.duplicate_summary = {
        "exact_duplicate_rows": exact_duplicate_count,
        "exact_duplicates_removed": exact_duplicate_count,
        "dates_with_conflicting_rows": duplicate_dates,
    }

    missing_by_asset: dict[str, int] = {}
    missing_by_date: list[dict[str, object]] = []
    missing_cells = 0
    invalid_prices: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        missing_assets: list[str] = []
        for asset in assets:
            raw_value = row.get(asset)
            if raw_value is None or not raw_value.strip():
                missing_by_asset[asset] = missing_by_asset.get(asset, 0) + 1
                missing_cells += 1
                missing_assets.append(asset)
                continue
            try:
                value = Decimal(raw_value.strip())
            except InvalidOperation:
                invalid_prices.append(
                    {
                        "date": row[fieldnames[0]],
                        "asset": asset,
                        "value": raw_value,
                        "issue": "non_numeric",
                    }
                )
                continue
            if not value.is_finite():
                invalid_prices.append(
                    {
                        "date": row[fieldnames[0]],
                        "asset": asset,
                        "value": raw_value,
                        "issue": "non_finite",
                    }
                )
            elif value <= 0:
                invalid_prices.append(
                    {
                        "date": row[fieldnames[0]],
                        "asset": asset,
                        "value": raw_value,
                        "issue": "non_positive",
                    }
                )
        if missing_assets:
            missing_by_date.append(
                {
                    "date": row[fieldnames[0]],
                    "missing_assets": missing_assets,
                    "missing_count": len(missing_assets),
                }
            )

    total_price_cells = len(rows) * len(assets)
    result.missingness = {
        "total_missing_cells": missing_cells,
        "percentage_missing": (100 * missing_cells / total_price_cells)
        if total_price_cells
        else 0.0,
        "missing_by_asset": {asset: missing_by_asset.get(asset, 0) for asset in assets},
        "dates_with_missing_observations": missing_by_date,
        "imputation_performed": False,
    }
    result.invalid_price_summary = {
        "invalid_price_count": len(invalid_prices),
        "invalid_prices": invalid_prices,
        "zero_count": sum(
            item["issue"] == "non_positive" and Decimal(str(item["value"])) == 0
            for item in invalid_prices
        ),
        "negative_count": sum(
            item["issue"] == "non_positive" and Decimal(str(item["value"])) < 0
            for item in invalid_prices
        ),
        "non_numeric_count": sum(
            item["issue"] == "non_numeric" for item in invalid_prices
        ),
        "non_finite_count": sum(
            item["issue"] == "non_finite" for item in invalid_prices
        ),
    }

    for asset in assets:
        valid_values: list[Decimal] = []
        asset_dates: list[str] = []
        missing_count = 0
        for row in rows:
            raw_value = row.get(asset)
            if raw_value is None or not raw_value.strip():
                missing_count += 1
                continue
            try:
                value = Decimal(raw_value.strip())
            except InvalidOperation:
                continue
            if value.is_finite() and value > 0:
                valid_values.append(value)
                asset_dates.append(str(row[fieldnames[0]]))
        metadata.append(
            {
                "ticker": asset,
                "first_date": asset_dates[0] if asset_dates else None,
                "last_date": asset_dates[-1] if asset_dates else None,
                "observation_count": len(asset_dates),
                "missing_count": missing_count,
                "missing_percentage": (100 * missing_count / len(rows))
                if rows
                else 0.0,
                "minimum_price": str(min(valid_values)) if valid_values else None,
                "maximum_price": str(max(valid_values)) if valid_values else None,
            }
        )
    result.asset_metadata = metadata

    if missing_cells:
        result.warnings.append(
            f"Found {missing_cells} missing price cell(s); none were imputed. "
            "Return generation is blocked until missing observations can be "
            "handled without fabrication."
        )
    if invalid_prices:
        result.errors.append(
            f"Found {len(invalid_prices)} invalid price observation(s)."
        )
    if len(assets) != 50:
        result.information.append(
            f"Dataset contains {len(assets)} assets; the project target is "
            "approximately 50. No assets were added or removed."
        )
    if not assets:
        result.status = "failed"
    if result.errors:
        result.status = "failed"
    elif result.warnings:
        result.status = "warning"
    if dates and not out_of_order and not calendar_gaps:
        result.information.append(
            "Observed date spacing contains no gaps over three calendar days."
        )
    result.rows = rows
    return result
