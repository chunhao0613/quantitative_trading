from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import load_df, save_df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # 1. Log Returns (Stationary)
    out["log_ret_1"] = np.log(out["close"] / out["close"].shift(1))
    out["log_ret_5"] = np.log(out["close"] / out["close"].shift(5))
    
    # 2. Moving Average Distances (Stationary instead of raw SMAs)
    for period in [5, 10, 20, 60]:
        sma = out["close"].rolling(period).mean()
        out[f"dist_sma_{period}"] = (out["close"] - sma) / sma
        
    # 3. Z-Score Standardization for MACD & RSI (Rolling)
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    # Rolling Z-score for MACD Histogram (window=60)
    roll_mean = macd_hist.rolling(60).mean()
    roll_std = macd_hist.rolling(60).std()
    out["macd_hist_z"] = (macd_hist - roll_mean) / (roll_std + 1e-8)
    
    out["rsi_14"] = _rsi(out["close"], period=14)
    # Rolling Z-score for RSI
    rsi_mean = out["rsi_14"].rolling(60).mean()
    rsi_std = out["rsi_14"].rolling(60).std()
    out["rsi_14_z"] = (out["rsi_14"] - rsi_mean) / (rsi_std + 1e-8)

    # 4. Volatility Features
    out["atr_14"] = _atr(out, period=14)
    out["atr_ratio"] = out["atr_14"] / out["close"]  # Normalized ATR
    out["vol_5"] = out["log_ret_1"].rolling(5).std()
    out["vol_20"] = out["log_ret_1"].rolling(20).std()
    
    return out


def run(args: argparse.Namespace) -> None:
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for clean_file in in_dir.glob("*_clean.parquet"):
        df = load_df(clean_file)
        feat = add_indicators(df)
        symbol = clean_file.name.split("_")[0]
        save_df(feat, out_dir / f"{symbol}_features.parquet")
        print(f"features: {symbol}, rows={len(feat)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add technical indicators")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed")
    run(parser.parse_args())
