from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.common.config import load_yaml
from src.etl.clean_align import clean_and_align
from src.etl.fetch_twse_tpex import fetch_market_data
from src.features.build_dataset import build_dataset
from src.features.technical_indicators import add_indicators
from src.models.train_eval import run as run_model
from src.backtest.run_backtest import run as run_backtest
from src.common.io_utils import save_df
from src.monitoring.metrics import FEATURE_COMPUTE_SECONDS, INFERENCE_LATENCY_MS, PIPELINE_LATENCY_MS


def run_pipeline(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    stocks = cfg["stocks"]
    data_source = cfg.get("data_source", "twse")
    lookback_years = int(cfg.get("lookback_years", 0))
    if lookback_years > 0:
        end_dt = date.today()
        start_dt = date(end_dt.year - lookback_years, end_dt.month, end_dt.day)
        start = start_dt.isoformat()
        end = end_dt.isoformat()
    else:
        start = cfg["start_date"]
        end = cfg["end_date"]
    allow_synthetic = bool(cfg.get("allow_synthetic", False))
    force_synthetic = bool(cfg.get("force_synthetic", False))

    raw_dir = ROOT / "data" / "raw"
    proc_dir = ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    for stock in stocks:
        raw = fetch_market_data(
            stock_no=stock,
            start=start,
            end=end,
            data_source=data_source,
            sleep_sec=0.05,
            allow_synthetic=allow_synthetic,
            force_synthetic=force_synthetic,
        )

        save_df(raw, raw_dir / f"{stock}_raw.parquet")

        clean = clean_and_align(raw, forward_fill=True)
        save_df(clean, proc_dir / f"{stock}_clean.parquet")

        with FEATURE_COMPUTE_SECONDS.time():
            feat = add_indicators(clean)
        save_df(feat, proc_dir / f"{stock}_features.parquet")

        ds = build_dataset(feat)
        save_df(ds, proc_dir / f"{stock}_dataset.parquet")

    m0 = time.perf_counter()
    run_model(
        argparse.Namespace(
            input_dir=str(proc_dir),
            report_dir=str(ROOT / "reports"),
            model_dir=str(ROOT / "artifacts"),
        )
    )
    INFERENCE_LATENCY_MS.observe((time.perf_counter() - m0) * 1000)

    run_backtest(
        argparse.Namespace(
            signals_dir=str(ROOT / "reports"),
            clean_dir=str(proc_dir),
            report_dir=str(ROOT / "reports"),
            initial_cash=1_000_000.0,
        )
    )

    PIPELINE_LATENCY_MS.set((time.perf_counter() - t0) * 1000)
    print("Phase 1 pipeline done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run phase 1 end-to-end pipeline")
    parser.add_argument("--config", default="configs/stocks.yaml")
    args = parser.parse_args()
    run_pipeline((ROOT / args.config).resolve())
