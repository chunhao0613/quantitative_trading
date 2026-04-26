# Quantitative Trading Backtest Infrastructure (PoC)

An end-to-end Python pipeline for quantitative factor research and backtesting, specifically designed for the Taiwan Stock Exchange (TWSE).

## System Architecture

This repository implements a modular pipeline encompassing Data ETL, Feature Engineering, Machine Learning Evaluation, and Event-driven Backtesting. 

- **ETL Pipeline:** Automated fetching and cleaning of OHLCV data via TWSE/TPEx APIs.
- **Factor Engineering:** Generation of stationary time-series features (Log Returns, Rolling Z-scored MACD/RSI) and volatility indicators (ATR).
- **Model Evaluation:** Regularized linear modeling (Ridge) with `TimeSeriesSplit` (gap=5) to prevent data leakage. Factor predictive power is evaluated using Information Coefficient (IC) and Rank IC.
- **Backtesting Engine:** Built on `Backtrader` with realistic friction models:
  - Commission: 0.1425%
  - Transaction Tax: 0.3% (Sell-side)
  - Slippage: Dynamic ATR-based modeling
- **Monitoring:** Integration with `prometheus_client` for backtest metric exposure.

## Project Structure

- `src/etl/`: Data extraction and alignment modules.
- `src/features/`: Feature generation and dataset building.
- `src/models/`: Model training, IC evaluation, and feature importance extraction.
- `src/backtest/`: Backtrader strategy definitions and execution.
- `src/monitoring/`: Prometheus metrics server.
- `scripts/`: Entry point scripts for pipeline execution.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Phase 1)
This will trigger ETL, Feature Engineering, Model Training, and Backtesting sequentially based on the targets defined in `configs/stocks_demo.yaml`.
```bash
python scripts/run_phase1.py --config configs/stocks_demo.yaml
```

### 3. Output Artifacts
All generated reports and performance charts are exported to the `reports/` directory, including:
- `model_report.md` (IC / Rank IC metrics)
- `backtest_report.md` (Sharpe Ratio, MDD, Returns)
- `*_equity_curve.png` (Visualized performance against Buy & Hold benchmark)