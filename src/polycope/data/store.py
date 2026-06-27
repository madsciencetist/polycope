"""Parquet-backed local data lake helpers.

We keep everything as Parquet on disk and query it with DuckDB when convenient.
No database server required; re-running experiments never re-hits the API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def exists(path: Path) -> bool:
    return Path(path).exists()
