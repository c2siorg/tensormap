"""Shared CSV header extraction and validation for DataFile records.

Upload-time caching and post-preprocessing cache refresh both need to read a
CSV's header without loading its rows; sharing this means they read it the
same way and treat a read failure the same way, instead of two independent
implementations quietly drifting apart.
"""

import csv

import pandas as pd


def read_csv_columns(file_path: str) -> list[str] | None:
    """Read a CSV's header without loading its rows, or None on failure."""
    try:
        return list(pd.read_csv(file_path, nrows=0).columns)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, OSError, UnicodeDecodeError, MemoryError):
        return None


def find_duplicate_columns(file_path: str) -> list[str]:
    """Inspect raw CSV headers and return duplicated column names in order of appearance.

    Surrounding whitespace is stripped before checking equality.
    Duplicate names are deduplicated in the returned list.
    """
    try:
        with open(file_path, encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and any(cell.strip() for cell in row):
                    seen: set[str] = set()
                    duplicates: list[str] = []
                    for cell in row:
                        name = cell.strip()
                        display_name = name or "(empty)"
                        if display_name in seen:
                            if display_name not in duplicates:
                                duplicates.append(display_name)
                        else:
                            seen.add(display_name)
                    return duplicates
            return []
    except (OSError, UnicodeDecodeError):
        return []
