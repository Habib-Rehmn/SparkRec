"""Strict partition-local parsing for MovieLens line-oriented CSV files."""

import csv
from collections.abc import Iterator
from pathlib import Path


def check_header(path: Path, columns: tuple[str, ...]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, strict=True), None)
    if header != list(columns):
        raise ValueError(f"{path.name}: expected header {','.join(columns)}, got {header}")


def parse_partition(index: int, lines: Iterator[str], width: int):
    """MovieLens uses one CSV record per line. Preserve malformed lines verbatim."""
    if index == 0:
        next(lines, None)  # Header was checked independently on the driver.
    for line in lines:
        try:
            values = next(csv.reader([line], strict=True))
            if len(values) != width:
                raise ValueError(f"expected {width} columns, got {len(values)}")
            yield (*values, line, None)
        except (csv.Error, ValueError) as exc:
            yield (*([None] * width), line, str(exc))
