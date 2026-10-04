"""Reproducible orchestration of the Phase 1 data pipeline."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from riskchallenger.data.loader import DEFAULT_RAW_PATH, load_prices, repository_root
from riskchallenger.data.metadata import write_metadata
from riskchallenger.data.returns import calculate_returns, validate_returns
from riskchallenger.data.validation import validate_prices
from riskchallenger.logging import get_logger

LOGGER = get_logger("data.pipeline")


def _write_wide_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str | float | None]],
    *,
    prices: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if prices:
                writer.writerow(row)
            else:
                writer.writerow(
                    {key: "" if value is None else value for key, value in row.items()}
                )


def run_data_pipeline(
    raw_path: Path | str | None = None,
    *,
    output_root: Path | str | None = None,
) -> dict[str, object]:
    """Validate source prices and write canonical prices, returns, metadata, and report.

    A report is written even when validation fails. Derived price/return outputs
    are written only when observations pass validation and contain no missingness.
    """
    root = repository_root()
    source_path = Path(raw_path) if raw_path is not None else root / DEFAULT_RAW_PATH
    if not source_path.is_absolute():
        source_path = root / source_path
    destination = Path(output_root) if output_root is not None else root
    if not destination.is_absolute():
        destination = root / destination
    LOGGER.info("Starting data validation for %s", source_path)

    fieldnames, source_records = load_prices(source_path)
    validation = validate_prices(fieldnames, source_records)
    LOGGER.info(
        "Validation %s: %d rows, %d assets",
        validation.status,
        len(validation.rows),
        len(validation.assets),
    )

    processed_dir = destination / "data/processed"
    report_path = destination / "data/reports/data_quality_report.json"
    prices_path = processed_dir / "nse50_prices_clean.csv"
    returns_path = processed_dir / "nse50_returns.csv"
    metadata_path = processed_dir / "nse50_metadata.csv"

    write_metadata(metadata_path, validation.asset_metadata)
    report: dict[str, object] = {
        "dataset_path": str(source_path),
        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": validation.status,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "information": validation.information,
        "schema": validation.schema,
        "row_count": len(validation.rows),
        "asset_count": len(validation.assets),
        "asset_names": validation.assets,
        "date_range": validation.date_summary,
        "duplicate_summary": validation.duplicate_summary,
        "missingness": validation.missingness,
        "invalid_price_summary": validation.invalid_price_summary,
        "asset_metadata": validation.asset_metadata,
        "return_dataset": {
            "status": "not_generated",
            "reason": "Price validation is incomplete.",
        },
        "output_paths": {
            "prices": str(prices_path),
            "returns": str(returns_path),
            "metadata": str(metadata_path),
        },
    }

    if validation.status != "failed" and not validation.missingness.get(
        "total_missing_cells", 0
    ):
        _write_wide_csv(prices_path, fieldnames, validation.rows, prices=True)
        return_rows = calculate_returns(
            validation.rows, validation.assets, fieldnames[0]
        )
        return_validation = validate_returns(
            validation.rows, return_rows, validation.assets, fieldnames[0]
        )
        report["return_dataset"] = return_validation
        if return_validation["status"] == "passed":
            _write_wide_csv(returns_path, fieldnames, return_rows)
            report["return_dataset"] = {
                **return_validation,
                "path": str(returns_path),
                "generated_row_count": len(return_rows),
                "generated_asset_count": len(validation.assets),
            }
            LOGGER.info("Wrote canonical prices to %s", prices_path)
            LOGGER.info("Wrote decimal returns to %s", returns_path)
        else:
            report["validation_status"] = "failed"
            report["errors"] = [*validation.errors, *return_validation["errors"]]
    else:
        LOGGER.warning("Data issues prevent canonical price and return generation")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote data quality report to %s", report_path)
    return report
