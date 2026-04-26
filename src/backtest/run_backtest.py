from __future__ import annotations

import argparse
import json
from pathlib import Path

import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.strategy_bt import MLSignalStrategy, SignalPandasData, TaiwanStockCommission


def run_single_backtest(signal_file: Path, clean_file: Path, initial_cash: float = 1_000_000.0) -> dict:
    signal_df = pd.read_parquet(signal_file)
    clean_df = pd.read_parquet(clean_file)

    merged = clean_df.merge(
        signal_df[["date", "signal", "atr_14"]],
        on="date",
        how="inner",
    ).dropna(subset=["open", "high", "low", "close", "volume"])

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").set_index("date")

    cerebro = bt.Cerebro(stdstats=False)
    data = SignalPandasData(dataname=merged)
    cerebro.adddata(data)
    cerebro.addstrategy(MLSignalStrategy)

    comminfo = TaiwanStockCommission(fee_rate=0.001425, tax_rate=0.003)
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.broker.setcash(initial_cash)

    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    results = cerebro.run()
    strat = results[0]

    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    timereturn = strat.analyzers.timereturn.get_analysis()

    equity = pd.Series(timereturn).sort_index()
    if not equity.empty:
        equity_curve = (1 + equity).cumprod() * initial_cash
        eq_df = pd.DataFrame({"date": equity_curve.index, "equity": equity_curve.values})
    else:
        eq_df = pd.DataFrame(columns=["date", "equity"])

    total_trades = int(trades.total.closed) if hasattr(trades, "total") and hasattr(trades.total, "closed") else 0
    won = int(trades.won.total) if hasattr(trades, "won") and hasattr(trades.won, "total") else 0
    win_rate = (won / total_trades) if total_trades > 0 else 0.0

    symbol = signal_file.name.split("_")[0]
    return {
        "symbol": symbol,
        "start_value": initial_cash,
        "end_value": float(cerebro.broker.getvalue()),
        "total_return": float(returns.get("rtot", 0.0)),
        "annual_return": float(returns.get("rnorm", 0.0)),
        "max_drawdown": float(drawdown.max.drawdown if hasattr(drawdown, "max") else 0.0),
        "sharpe": float(sharpe.get("sharperatio", 0.0) or 0.0),
        "total_trades": total_trades,
        "win_rate": win_rate,
        "fee_rate": 0.001425,
        "tax_rate": 0.003,
        "slippage_model": "dynamic_atr",
        "equity_curve": eq_df,
        "price_series": merged[["close"]].reset_index(),
    }


def run(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    signals_dir = Path(args.signals_dir)
    clean_dir = Path(args.clean_dir)

    for signal_file in signals_dir.glob("*_signals.parquet"):
        symbol = signal_file.name.split("_")[0]
        clean_file = clean_dir / f"{symbol}_clean.parquet"
        if not clean_file.exists():
            continue
        summary = run_single_backtest(signal_file=signal_file, clean_file=clean_file, initial_cash=args.initial_cash)
        summaries.append(summary)
        print(f"backtest: {symbol} return={summary['total_return']:.4f} sharpe={summary['sharpe']:.4f}")

        if not summary["equity_curve"].empty:
            eq = summary["equity_curve"].copy()
            px = summary["price_series"].copy()
            eq["date"] = pd.to_datetime(eq["date"])
            px["date"] = pd.to_datetime(px["date"])
            px["price_norm"] = px["close"] / px["close"].iloc[0] * args.initial_cash

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(eq["date"], eq["equity"], label="Equity Curve", linewidth=1.8)
            ax.plot(px["date"], px["price_norm"], label="Price (Normalized)", alpha=0.7)
            ax.set_title(f"{symbol} Price vs Equity Curve")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(report_dir / f"{symbol}_equity_curve.png", dpi=140)
            plt.close(fig)

            eq.to_csv(report_dir / f"{symbol}_equity_curve.csv", index=False)

    serializable = []
    for s in summaries:
        row = dict(s)
        row.pop("equity_curve", None)
        row.pop("price_series", None)
        serializable.append(row)

    with (report_dir / "backtest_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    lines = ["# Backtest Report", "", "成本假設：手續費 0.001425、證交稅 0.003、ATR 動態滑價。", ""]
    for s in serializable:
        lines.extend(
            [
                f"## {s['symbol']}",
                f"- total_return: {s['total_return']:.4f}",
                f"- annual_return: {s['annual_return']:.4f}",
                f"- sharpe: {s['sharpe']:.4f}",
                f"- max_drawdown: {s['max_drawdown']:.4f}",
                f"- total_trades: {s['total_trades']}",
                f"- win_rate: {s['win_rate']:.4f}",
                f"- fee_rate: {s['fee_rate']}",
                f"- tax_rate: {s['tax_rate']}",
                f"- slippage_model: {s['slippage_model']}",
                f"- equity_curve_chart: {s['symbol']}_equity_curve.png",
                "",
            ]
        )

    (report_dir / "backtest_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest with transaction costs and slippage")
    parser.add_argument("--signals-dir", default="reports")
    parser.add_argument("--clean-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    run(parser.parse_args())
