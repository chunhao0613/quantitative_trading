from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.io_utils import load_df, save_df

FEATURE_COLS = [
    "log_ret_1",
    "log_ret_5",
    "dist_sma_5",
    "dist_sma_10",
    "dist_sma_20",
    "dist_sma_60",
    "macd_hist_z",
    "rsi_14_z",
    "atr_ratio",
    "vol_5",
    "vol_20",
]


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")

    for col in FEATURE_COLS:
        out[col] = out[col].shift(1)

    # Compute forward returns for predictive targets
    out["fwd_ret_1"] = out["close"].shift(-1) / out["close"] - 1.0
    out["fwd_ret_5"] = out["close"].shift(-5) / out["close"] - 1.0
    
    # We will use 1-day forward return as the primary target for continuous prediction
    out["target_ret"] = out["fwd_ret_1"]
    
    # Drop rows where target or features are NA
    out = out.dropna(subset=FEATURE_COLS + ["target_ret"])
    return out


def run(args: argparse.Namespace) -> None:
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_dict = {"feature_columns": FEATURE_COLS}

    for feature_file in in_dir.glob("*_features.parquet"):
        df = load_df(feature_file)
        ds = build_dataset(df)
        symbol = feature_file.name.split("_")[0]
        save_df(ds, out_dir / f"{symbol}_dataset.parquet")
        print(f"dataset: {symbol}, rows={len(ds)}")

    with (out_dir / "feature_dict.json").open("w", encoding="utf-8") as f:
        json.dump(feature_dict, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ML dataset")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed")
    run(parser.parse_args())
