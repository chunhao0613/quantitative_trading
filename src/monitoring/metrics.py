from __future__ import annotations

from prometheus_client import Gauge, Histogram

PIPELINE_LATENCY_MS = Gauge("pipeline_latency_ms", "End-to-end pipeline latency in milliseconds")
FEATURE_COMPUTE_SECONDS = Histogram("feature_compute_seconds", "Feature computation duration seconds")
INFERENCE_LATENCY_MS = Histogram("inference_latency_ms", "Model inference latency milliseconds")
BACKTEST_RUNTIME_SECONDS = Gauge("backtest_runtime_seconds", "Backtest runtime in seconds")
MEMORY_USAGE_MB = Gauge("memory_usage_mb", "Process memory usage in MB")
CPU_USAGE_PERCENT = Gauge("cpu_usage_percent", "Process CPU usage percent")
MODEL_F1 = Gauge("model_f1", "Model F1 score")
MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy")
BACKTEST_RETURN = Gauge("backtest_return", "Backtest total return")
BACKTEST_WIN_RATE = Gauge("backtest_win_rate", "Backtest win rate")
