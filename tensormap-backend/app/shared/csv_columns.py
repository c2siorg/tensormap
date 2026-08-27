"""Shared CSV header extraction for the DataFile.columns cache.

Upload-time caching and post-preprocessing cache refresh both need to read a
CSV's header without loading its rows; sharing this means they read it the
same way and treat a read failure the same way, instead of two independent
implementations quietly drifting apart.
"""

import pandas as pd


def read_csv_columns(file_path: str) -> list[str] | None:
    """Read a CSV's header without loading its rows, or None on failure."""
    try:
        return list(pd.read_csv(file_path, nrows=0).columns)
    except (pd.errors.ParserError, OSError, UnicodeDecodeError, MemoryError):
        return None
