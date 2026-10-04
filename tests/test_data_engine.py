"""Unit and integration tests for the Phase 1 data engine."""

import csv
import json
from pathlib import Path

import pytest

from riskchallenger.data.loader import load_prices
from riskchallenger.data.pipeline import run_data_pipeline
from riskchallenger.data.returns import calculate_returns, validate_returns
from riskchallenger.data.validation import validate_prices


def write_csv(path: Path, rows: list[list[str]], header: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def test_loader_reads_valid_dataset(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    write_csv(path, [["2024-01-02", "100"]], ["date", "ABC.NS"])
    fields, records = load_prices(path)
    assert fields == ["date", "ABC.NS"]
    assert records == [{"date": "2024-01-02", "ABC.NS": "100"}]


def test_loader_missing_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Price dataset not found"):
        load_prices(tmp_path / "missing.csv")


def test_loader_rejects_duplicate_headers(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("date,ABC,ABC\n2024-01-02,10,11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column"):
        load_prices(path)


def test_valid_wide_schema_passes() -> None:
    result = validate_prices(
        ["date", "ABC.NS"], [{"date": "2024-01-02", "ABC.NS": "100"}]
    )
    assert result.status == "passed"
    assert result.schema["format"] == "wide"
    assert result.assets == ["ABC.NS"]


def test_malformed_schema_fails() -> None:
    result = validate_prices(["timestamp", "ABC.NS"], [])
    assert result.status == "failed"
    assert "date field" in result.errors[0]


def test_invalid_date_is_reported_without_exception() -> None:
    result = validate_prices(
        ["date", "ABC.NS"], [{"date": "not-a-date", "ABC.NS": "100"}]
    )
    assert result.status == "failed"
    assert result.date_summary["invalid_dates"] == [{"row": 2, "value": "not-a-date"}]


def test_out_of_order_rows_are_sorted() -> None:
    result = validate_prices(
        ["date", "ABC.NS"],
        [
            {"date": "2024-01-03", "ABC.NS": "105"},
            {"date": "2024-01-02", "ABC.NS": "100"},
        ],
    )
    assert result.status == "passed"
    assert [row["date"] for row in result.rows] == ["2024-01-02", "2024-01-03"]
    assert result.date_summary["input_was_chronological"] is False


def test_exact_duplicates_removed_and_conflicts_fail() -> None:
    duplicate = {"date": "2024-01-02", "ABC.NS": "100"}
    result = validate_prices(["date", "ABC.NS"], [duplicate, duplicate.copy()])
    assert result.status == "passed"
    assert len(result.rows) == 1
    assert result.duplicate_summary["exact_duplicates_removed"] == 1

    conflict = validate_prices(
        ["date", "ABC.NS"],
        [duplicate, {"date": "2024-01-02", "ABC.NS": "101"}],
    )
    assert conflict.status == "failed"
    assert conflict.duplicate_summary["dates_with_conflicting_rows"] == ["2024-01-02"]


def test_missing_values_are_detected_and_never_imputed() -> None:
    result = validate_prices(["date", "ABC.NS"], [{"date": "2024-01-02", "ABC.NS": ""}])
    assert result.status == "warning"
    assert result.missingness["total_missing_cells"] == 1
    assert result.missingness["imputation_performed"] is False
    assert result.rows[0]["ABC.NS"] == ""


@pytest.mark.parametrize(
    "price,issue",
    [
        ("0", "non_positive"),
        ("-1", "non_positive"),
        ("NaN", "non_finite"),
        ("inf", "non_finite"),
        ("bad", "non_numeric"),
    ],
)
def test_invalid_prices_are_reported(price: str, issue: str) -> None:
    result = validate_prices(
        ["date", "ABC.NS"], [{"date": "2024-01-02", "ABC.NS": price}]
    )
    assert result.status == "failed"
    assert result.invalid_price_summary["invalid_prices"][0]["issue"] == issue


def test_positive_prices_pass() -> None:
    result = validate_prices(
        ["date", "ABC.NS"], [{"date": "2024-01-02", "ABC.NS": "100.00"}]
    )
    assert result.status == "passed"
    assert result.rows[0]["ABC.NS"] == "100.00"


def test_returns_follow_simple_return_formula_and_are_decimal() -> None:
    prices = [
        {"date": "2024-01-02", "ABC.NS": "100"},
        {"date": "2024-01-03", "ABC.NS": "105"},
    ]
    returns = calculate_returns(prices, ["ABC.NS"])
    assert returns[0]["ABC.NS"] is None
    assert returns[1]["ABC.NS"] == pytest.approx(0.05)
    assert returns[1]["ABC.NS"] != 5
    assert validate_returns(prices, returns, ["ABC.NS"])["status"] == "passed"


def test_returns_reject_missing_price() -> None:
    with pytest.raises(ValueError, match="missing price"):
        calculate_returns([{"date": "2024-01-02", "ABC.NS": ""}], ["ABC.NS"])


def test_actual_dataset_pipeline_outputs(tmp_path: Path) -> None:
    report = run_data_pipeline(output_root=tmp_path)
    processed = tmp_path / "data/processed"
    expected = [
        processed / "nse50_prices_clean.csv",
        processed / "nse50_returns.csv",
        processed / "nse50_metadata.csv",
        tmp_path / "data/reports/data_quality_report.json",
    ]
    assert all(path.is_file() for path in expected)
    assert report["validation_status"] == "passed"
    assert report["row_count"] > 0
    assert report["asset_count"] == 50
    with expected[0].open(newline="", encoding="utf-8") as source:
        prices = list(csv.DictReader(source))
    with expected[1].open(newline="", encoding="utf-8") as source:
        returns = list(csv.DictReader(source))
    with expected[2].open(newline="", encoding="utf-8") as source:
        metadata = list(csv.DictReader(source))
    saved_report = json.loads(expected[3].read_text(encoding="utf-8"))
    assert len(prices) == len(returns) == report["row_count"]
    assert len(metadata) == 50
    asset = "HDFCBANK.NS"
    expected_return = float(prices[1][asset]) / float(prices[0][asset]) - 1
    assert float(returns[1][asset]) == pytest.approx(expected_return)
    assert returns[0][asset] == ""
    assert saved_report["return_dataset"]["status"] == "passed"
