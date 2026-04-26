from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
from prometheus_client import start_http_server

from src.monitoring.metrics import (
    BACKTEST_RETURN,
    BACKTEST_RUNTIME_SECONDS,
    BACKTEST_WIN_RATE,
    CPU_USAGE_PERCENT,
    MEMORY_USAGE_MB,
    MODEL_ACCURACY,
    MODEL_F1,
)


def update_from_reports(report_dir: Path) -> None:
    model_json = report_dir / "model_metrics.json"
    if model_json.exists():
        rows = json.loads(model_json.read_text(encoding="utf-8"))
        if rows:
            avg_acc = sum(r["accuracy"] for r in rows) / len(rows)
            avg_f1 = sum(r["f1"] for r in rows) / len(rows)
            MODEL_ACCURACY.set(avg_acc)
            MODEL_F1.set(avg_f1)

    backtest_json = report_dir / "backtest_metrics.json"
    if backtest_json.exists():
        rows = json.loads(backtest_json.read_text(encoding="utf-8"))
        if rows:
            avg_ret = sum(r["total_return"] for r in rows) / len(rows)
            avg_win = sum(r["win_rate"] for r in rows) / len(rows)
            BACKTEST_RETURN.set(avg_ret)
            BACKTEST_WIN_RATE.set(avg_win)


def run(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    start_http_server(args.port)
    print(f"Prometheus metrics on :{args.port}/metrics")

    proc = psutil.Process()
    while True:
        t0 = time.perf_counter()
        update_from_reports(report_dir)
        MEMORY_USAGE_MB.set(proc.memory_info().rss / 1024 / 1024)
        CPU_USAGE_PERCENT.set(proc.cpu_percent(interval=0.1))
        BACKTEST_RUNTIME_SECONDS.set(time.perf_counter() - t0)
        time.sleep(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Prometheus metrics server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--report-dir", default="reports")
    run(parser.parse_args())
