"""Convenience loaders that return (trades, markets) from either source."""

from __future__ import annotations

import pandas as pd

from .config import settings
from .data.store import read_parquet
from .features.trades import add_duration_bucket
from .synth import generate


def load_dataset(synthetic: bool = False, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trades, markets). Synthetic data, or the ingested Parquet lake."""
    if synthetic:
        trades, markets, _ = generate(seed=seed)
        return trades, markets

    trades = read_parquet(settings.raw_dir / "trades.parquet")
    markets = read_parquet(settings.raw_dir / "markets.parquet")
    if "duration_bucket" not in markets or markets["duration_bucket"].eq("").all():
        markets = add_duration_bucket(markets)
    return trades, markets
