"""CSV loading for the repository's source price dataset."""

import csv
from pathlib import Path

from riskchallenger.logging import get_logger

LOGGER = get_logger("data.loader")
DEFAULT_RAW_PATH = Path("data/raw/dataset_prices.csv")


def repository_root() -> Path:
    """Return the project root based on this installed source package."""
    return Path(__file__).resolve().parents[3]


def load_prices(
    path: Path | str | None = None,
) -> tuple[list[str], list[dict[str, str | None]]]:
    """Load a wide price CSV without changing or dropping source values."""
    source_path = (
        Path(path) if path is not None else repository_root() / DEFAULT_RAW_PATH
    )
    if not source_path.is_absolute():
        source_path = repository_root() / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Price dataset not found: {source_path}")

    LOGGER.info("Loading source price dataset: %s", source_path)
    try:
        with source_path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError(f"Dataset has no header row: {source_path}")
            if len(fieldnames) != len(set(fieldnames)):
                raise ValueError("Dataset contains duplicate column names")
            if any(not name or not name.strip() for name in fieldnames):
                raise ValueError("Dataset contains an empty column name")

            records: list[dict[str, str | None]] = []
            for line_number, record in enumerate(reader, start=2):
                if None in record:
                    raise ValueError(
                        f"Row {line_number} has more fields than the header"
                    )
                records.append(record)
    except UnicodeDecodeError as error:
        raise ValueError(f"Dataset is not valid UTF-8: {source_path}") from error
    LOGGER.info("Loaded %d rows and %d columns", len(records), len(fieldnames))
    return fieldnames, records
