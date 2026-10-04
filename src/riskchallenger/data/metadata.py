"""Serialization of per-asset descriptive metadata."""

import csv
from pathlib import Path

METADATA_FIELDS = [
    "ticker",
    "first_date",
    "last_date",
    "observation_count",
    "missing_count",
    "missing_percentage",
    "minimum_price",
    "maximum_price",
]


def write_metadata(path: Path, metadata: list[dict[str, object]]) -> None:
    """Write per-asset coverage and price-range metadata as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(metadata)
