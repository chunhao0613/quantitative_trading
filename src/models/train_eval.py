from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.common.io_utils import load_df
from src.features.build_dataset import FEATURE_COLS

def calculate_ic(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0, 0.0
    ic, _ = pearsonr(y_true, y_pred)
    ric, _ = spearmanr(y_true, y_pred)
    return float(ic), float(ric)

def evaluate_symbol(dataset_path: Path, report_dir: Path, model_dir: Path) -> dict:
    df = load_df(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if len(df) < 60:
        symbol = dataset_path.name.split("_")[0]
        return {
            "symbol": symbol,
            "rows": int(len(df)),
            "skipped": True,
            "reason": "insufficient_rows",
        }

    X = df[FEATURE_COLS]
    y = df["target_ret"].astype(float) # Continuous return

    # GapTimeSeriesSplit to prevent look-ahead bias and reduce overfitting
    # We leave a gap of 5 days (assuming fwd_ret_5 is maximum target)
    gap = 5
    tscv = TimeSeriesSplit(n_splits=3, gap=gap)
    fold_rows = []

    # Use a regularized linear model instead of RF for better interpretability and robustness
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", Ridge(alpha=100.0, random_state=42)),
        ]
    )

    oof_pred = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        oof_pred[test_idx] = pred

        ic, ric = calculate_ic(y_test, pred)

        fold_rows.append(
            {
                "fold": fold,
                "ic": ic,
                "rank_ic": ric,
            }
        )

    valid = ~np.isnan(oof_pred)
    y_true = y[valid]
    y_pred = oof_pred[valid]

    symbol = dataset_path.name.split("_")[0]
    
    overall_ic, overall_ric = calculate_ic(y_true, y_pred)
    
    final_metrics = {
        "symbol": symbol,
        "rows": int(len(df)),
        "ic": overall_ic,
        "rank_ic": overall_ric,
        "fold_metrics": fold_rows,
    }

    # Out-of-sample holdout test (e.g., last 20%)
    train_size = int(len(df) * 0.8)
    X_train_hold = X.iloc[:train_size]
    y_train_hold = y.iloc[:train_size]
    X_test_hold = X.iloc[train_size+gap:]
    y_test_hold = y.iloc[train_size+gap:]

    if len(X_test_hold) > 10:
        pipe.fit(X_train_hold, y_train_hold)
        hold_pred = pipe.predict(X_test_hold)
        hold_ic, hold_ric = calculate_ic(y_test_hold, hold_pred)
        
        final_metrics["holdout"] = {
            "train_rows": int(len(X_train_hold)),
            "test_rows": int(len(X_test_hold)),
            "ic": hold_ic,
            "rank_ic": hold_ric,
        }

    # Fit final model on all rows for producing backtest signals.
    pipe.fit(X, y)
    joblib.dump(pipe, model_dir / f"{symbol}_ridge.joblib")
    
    # Feature Importance (XAI)
    # For Ridge, standardized coefficients represent feature importance
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": coefs
    }).sort_values("importance", key=abs, ascending=False)
    
    # Save to CSV
    importance_df.to_csv(report_dir / f"{symbol}_feature_importance.csv", index=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color='skyblue')
    plt.title(f"Feature Importances (Ridge Coefficients) for {symbol}")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(report_dir / f"{symbol}_feature_importance.png")
    plt.close()

    preds = pd.DataFrame(
        {
            "date": df["date"],
            "symbol": df["symbol"],
            "close": df["close"],
            "signal": pipe.predict(X), # Continuous signal
            "atr_14": df["atr_14"],
        }
    )
    preds.to_parquet(report_dir / f"{symbol}_signals.parquet", index=False)

    with (report_dir / f"{symbol}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    return final_metrics


def run(args: argparse.Namespace) -> None:
    in_dir = Path(args.input_dir)
    report_dir = Path(args.report_dir)
    model_dir = Path(args.model_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for dataset in in_dir.glob("*_dataset.parquet"):
        metrics = evaluate_symbol(dataset, report_dir=report_dir, model_dir=model_dir)
        all_metrics.append(metrics)
        if metrics.get("skipped"):
            print(f"model: {metrics['symbol']} skipped ({metrics['reason']}, rows={metrics['rows']})")
        else:
            print(f"model: {metrics['symbol']} IC={metrics['ic']:.4f} Rank_IC={metrics['rank_ic']:.4f}")

    with (report_dir / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    lines = ["# Quantitative Model Report (Information Coefficient)", ""]
    for m in all_metrics:
        if m.get("skipped"):
            lines.extend(
                [
                    f"## {m['symbol']}",
                    f"- rows: {m['rows']}",
                    f"- status: skipped",
                    f"- reason: {m['reason']}",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"## {m['symbol']}",
                f"- rows: {m['rows']}",
                f"- IC: {m['ic']:.4f}",
                f"- Rank IC: {m['rank_ic']:.4f}",
            ]
        )
        if "holdout" in m:
            lines.append(f"- Holdout IC: {m['holdout']['ic']:.4f}")
            lines.append(f"- Holdout Rank IC: {m['holdout']['rank_ic']:.4f}")
        lines.append("")

    (report_dir / "model_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate regression factor model")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--model-dir", default="artifacts")
    run(parser.parse_args())
