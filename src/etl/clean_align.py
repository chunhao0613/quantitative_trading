from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import load_df, save_df


def clean_and_align(df: pd.DataFrame, forward_fill: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date", "symbol"])

    numeric_cols = ["open", "high", "low", "close", "volume", "turnover", "transactions", "change"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    start, end = out["date"].min(), out["date"].max()
    all_days = pd.bdate_range(start=start, end=end)

    aligned = (
        out.set_index("date")
        .reindex(all_days)
        .rename_axis("date")
        .reset_index()
    )
    aligned["symbol"] = aligned["symbol"].ffill().bfill()
    aligned["is_trading_day"] = aligned["close"].notna().astype(int)

    if forward_fill:
        for c in ["open", "high", "low", "close"]:
            aligned[c] = aligned[c].ffill()
        aligned["volume"] = aligned["volume"].fillna(0)
        aligned["turnover"] = aligned["turnover"].fillna(0)
        aligned["transactions"] = aligned["transactions"].fillna(0)
        aligned["change"] = aligned["change"].fillna(0)

    return aligned


def run(args: argparse.Namespace) -> None:
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in in_dir.glob("*_raw.parquet"):
        df = load_df(raw_file)
        out = clean_and_align(df, forward_fill=args.forward_fill)
        symbol = raw_file.name.split("_")[0]
        save_df(out, out_dir / f"{symbol}_clean.parquet")
        print(f"cleaned: {symbol}, rows={len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and align time series")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--forward-fill", action="store_true")
    run(parser.parse_args())
