# Quantitative Trading Phase 1

本專案實作第一階段可重現流程：
- 台股歷史日 K 抓取與清洗
- 技術特徵工程（RSI、MACD、Bollinger、SMA、ATR）
- ML 預測隔日漲跌（RandomForest + TimeSeriesSplit）
- Backtrader 回測（手續費 0.001425、證交稅 0.003、動態滑價）
- Prometheus 監控端點

## 專案結構
- `src/etl`: 資料抓取/清洗
- `src/features`: 指標與資料集
- `src/models`: 訓練與評估
- `src/backtest`: 回測
- `src/monitoring`: 指標埋點
- `scripts/run_phase1.py`: 一鍵執行
- `docs/phase1_methodology.md`: 方法與金融知識說明
- `docs/jd_coverage_matrix.md`: JD 對照

## 快速開始（本機）
```powershell
pip install -r requirements.txt
python scripts/run_phase1.py --config configs/stocks_demo.yaml
```

真實資料模式（TWSE API 為主，yfinance 為 fallback，抓過去 3 年）：
```powershell
python scripts/run_phase1.py --config configs/stocks.yaml
```

## `stocks.yaml` 與 `stocks_demo.yaml` 差異
- `configs/stocks.yaml`:
	- `data_source: twse`
	- `force_synthetic: false`
	- 以 TWSE API 為主，抓 3 年日 K，失敗時會切到 yfinance 備援。
- `configs/stocks_demo.yaml`:
	- `data_source: twse`
	- `force_synthetic: true`
	- 強制使用合成資料，結果可重現，適合 demo 或離線環境。

兩者其他欄位相同（股票清單、lookback 年數、fallback 行為）。

完成後可查看：
- `reports/model_report.md`
- `reports/backtest_report.md`
- `reports/model_metrics.json`
- `reports/backtest_metrics.json`

## 啟動監控
```powershell
python -m src.monitoring.server --port 8000 --report-dir reports
```

查看 metrics：
```powershell
Invoke-WebRequest http://localhost:8000/metrics
```

## Docker
```powershell
docker build -t qt-phase1 .
docker run --rm -p 8000:8000 qt-phase1
```

## 驗證（測試）
```powershell
pytest -q
```

## 重要聲明
- 本專案提供可重現與可驗證的研究流程，不保證未來市場準確率或獲利。
- 任何策略上線前，需持續做窗口外驗證與風險控管。

## Notebook 與 PDF
Notebook 模板：`notebooks/quant_report_template.ipynb`

匯出 PDF（需本機有 Jupyter 與 LaTeX 環境）：
```powershell
jupyter nbconvert --to pdf notebooks/quant_report_template.ipynb
```
