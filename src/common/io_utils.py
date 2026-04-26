from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
