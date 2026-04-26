from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.common.io_utils import save_df

TWSE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"


def _to_yf_symbol(symbol: str) -> str:
    if symbol.endswith(".TW") or symbol.endswith(".TWO"):
        return symbol
    if symbol.isdigit():
        return f"{symbol}.TW"
    return symbol


def fetch_yfinance_range(stock_no: str, start: str, end: str) -> pd.DataFrame:
    yf_symbol = _to_yf_symbol(stock_no)
    data = yf.download(
        tickers=yf_symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    data = data.rename(columns=str.lower).reset_index()
    data = data.rename(columns={"date": "date"})
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in data.columns:
            data[col] = np.nan

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(data["Date"] if "Date" in data.columns else data["date"]),
            "volume": pd.to_numeric(data["volume"], errors="coerce").fillna(0),
            "turnover": 0.0,
            "open": pd.to_numeric(data["open"], errors="coerce"),
            "high": pd.to_numeric(data["high"], errors="coerce"),
            "low": pd.to_numeric(data["low"], errors="coerce"),
            "close": pd.to_numeric(data["close"], errors="coerce"),
            "change": pd.to_numeric(data["close"], errors="coerce").diff().fillna(0),
            "transactions": 0,
            "symbol": stock_no,
        }
    )
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _roc_to_ad(roc_date: str) -> pd.Timestamp:
    year, month, day = roc_date.split("/")
    return pd.Timestamp(int(year) + 1911, int(month), int(day))


def _parse_twse_json(payload: dict) -> pd.DataFrame:
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()

    columns = [
        "date",
        "volume",
        "turnover",
        "open",
        "high",
        "low",
        "close",
        "change",
        "transactions",
    ]
    out = []
    for row in rows:
        out.append(
            {
                "date": _roc_to_ad(row[0]),
                "volume": row[1].replace(",", ""),
                "turnover": row[2].replace(",", ""),
                "open": row[3].replace(",", ""),
                "high": row[4].replace(",", ""),
                "low": row[5].replace(",", ""),
                "close": row[6].replace(",", ""),
                "change": row[7].replace(",", ""),
                "transactions": row[8].replace(",", ""),
            }
        )
    df = pd.DataFrame(out, columns=columns)
    for c in columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_twse_month(stock_no: str, year: int, month: int, timeout: int = 8) -> pd.DataFrame:
    params = {
        "response": "json",
        "date": f"{year}{month:02d}01",
        "stockNo": stock_no,
    }
    response = requests.get(TWSE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return _parse_twse_json(payload)


def _generate_synthetic(stock_no: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    idx = pd.bdate_range(start=start_ts, end=end_ts)
    seed = int(stock_no) if stock_no.isdigit() else 42
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0003, scale=0.015, size=len(idx))
    price = 100 * np.exp(np.cumsum(steps))
    close = pd.Series(price, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, len(idx)))
    high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + rng.uniform(0.0, 0.01, len(idx)))
    low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - rng.uniform(0.0, 0.01, len(idx)))
    volume = rng.integers(5_000_000, 30_000_000, len(idx))
    turnover = volume * close.values
    transactions = rng.integers(3_000, 30_000, len(idx))

    syn = pd.DataFrame(
        {
            "date": idx,
            "volume": volume,
            "turnover": turnover,
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "change": close.diff().fillna(0).values,
            "transactions": transactions,
            "symbol": stock_no,
        }
    )
    return syn.reset_index(drop=True)


def fetch_twse_range(
    stock_no: str,
    start: str,
    end: str,
    sleep_sec: float = 0.4,
    allow_synthetic: bool = False,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if force_synthetic:
        return _generate_synthetic(stock_no, start_ts=start_ts, end_ts=end_ts)
    months = pd.period_range(start=start_ts, end=end_ts, freq="M")

    frames: list[pd.DataFrame] = []
    fail_count = 0
    for p in months:
        try:
            frame = fetch_twse_month(stock_no=stock_no, year=p.year, month=p.month)
            if not frame.empty:
                frames.append(frame)
            fail_count = 0
        except Exception:
            fail_count += 1
            if allow_synthetic and fail_count >= 2 and not frames:
                return _generate_synthetic(stock_no, start_ts=start_ts, end_ts=end_ts)
            if not allow_synthetic:
                raise
        time.sleep(sleep_sec)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        df["symbol"] = stock_no
        return df.reset_index(drop=True)

    if not allow_synthetic:
        raise RuntimeError(f"No data fetched for {stock_no}")

    return _generate_synthetic(stock_no, start_ts=start_ts, end_ts=end_ts)


def fetch_market_data(
    stock_no: str,
    start: str,
    end: str,
    data_source: str = "twse",
    sleep_sec: float = 0.4,
    allow_synthetic: bool = False,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    source = data_source.lower()
    if force_synthetic:
        return _generate_synthetic(stock_no, start_ts=pd.Timestamp(start), end_ts=pd.Timestamp(end))

    if source == "yfinance":
        df = fetch_yfinance_range(stock_no=stock_no, start=start, end=end)
        if not df.empty:
            return df
        if allow_synthetic:
            return _generate_synthetic(stock_no, start_ts=pd.Timestamp(start), end_ts=pd.Timestamp(end))
        raise RuntimeError(f"No yfinance data for {stock_no}")

    try:
        return fetch_twse_range(
            stock_no=stock_no,
            start=start,
            end=end,
            sleep_sec=sleep_sec,
            allow_synthetic=allow_synthetic,
            force_synthetic=force_synthetic,
        )
    except Exception:
        if source == "twse":
            fallback = fetch_yfinance_range(stock_no=stock_no, start=start, end=end)
            if not fallback.empty:
                fallback["symbol"] = stock_no
                return fallback.reset_index(drop=True)
        if allow_synthetic:
            return _generate_synthetic(stock_no, start_ts=pd.Timestamp(start), end_ts=pd.Timestamp(end))
        raise


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stock in args.stocks:
        df = fetch_market_data(
            stock_no=stock,
            start=args.start,
            end=args.end,
            data_source=args.data_source,
            allow_synthetic=args.allow_synthetic,
        )
        save_df(df, out_dir / f"{stock}_raw.parquet")
        print(f"saved: {stock}, rows={len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TWSE daily OHLCV")
    parser.add_argument("--stocks", nargs="+", required=True, help="Stock symbols, e.g. 2317 2618")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--data-source", default="twse", choices=["twse", "yfinance"])
    parser.add_argument("--allow-synthetic", action="store_true")
    run(parser.parse_args())
