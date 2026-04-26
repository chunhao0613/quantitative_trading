# Phase 1: Methodology & Architecture Design

This document outlines the engineering and quantitative research methodology implemented in the current PoC.

## 1. Data Processing & Feature Engineering
To address the non-stationarity of raw financial data, the following transformations are applied before model ingestion:
- **Stationary Returns:** Replaced absolute price features with 1-day and 5-day Log Returns.
- **Rolling Standardization:** Applied 60-day rolling Z-score normalization to momentum indicators (MACD, RSI) to maintain cross-sectional comparability.
- **Volatility Scaling:** Incorporated 14-day Average True Range (ATR) and historical return volatility as dynamic risk features.

## 2. Model Training & Validation
- **Algorithm Selection:** Selected `Ridge` regression (alpha=100.0) over non-linear tree-based models to mitigate overfitting on noisy financial data and ensure linear interpretability.
- **Cross-Validation:** Implemented `TimeSeriesSplit` with a 5-day gap to strictly prevent look-ahead bias and label leakage.
- **Evaluation Metrics:** Replaced simple Accuracy with the Information Coefficient (IC) and Rank IC to assess the true ordinal predictive power of the extracted factors. Out-of-sample (Holdout) tests are configured on the latest 20% of the dataset.
- **Explainability (XAI):** Standardized regression coefficients are extracted and visualized as Feature Importance to validate market intuition.

## 3. Backtesting Strictness
To avoid survivorship bias and unrealistic performance (laboratory logic), the `Backtrader` engine is configured with local market constraints:
- **Friction Costs:** Hard-coded TWSE standard rates (0.001425 commission, 0.003 tax on sell).
- **Signal Execution:** Signals are generated on `T` close and executed on `T+1` open.
- **Slippage Constraint:** Executions are subjected to an ATR-based dynamic slippage model to simulate liquidity impact.

## 4. Next Steps (Phase 2 Roadmap)
- Integrate LLM and RAG architecture to parse unstructured financial news and forum texts, extracting Sentiment Scores as orthogonal Alpha factors.
- Containerize the complete pipeline (`Docker`) and deploy long-term tracking dashboards via `Grafana`.