# Phase 1 方法論與知識說明

## 專案目標
建立可重現的台股量化研究流程：資料抓取、清洗對齊、技術指標特徵、機器學習預測、嚴謹回測、監控埋點。

## 技術棧與用途
- Python: 主開發語言，整合 ETL、ML、回測與監控。
- pandas / NumPy: 時間序列資料處理與數值運算。
- scikit-learn: 分類模型、時序交叉驗證、評估指標。
- backtrader: 回測引擎，納入交易成本與滑價。
- prometheus_client: 監控端點與系統健康度指標。
- Docker: 環境重現與工程協作。

## 流程與做法
1. ETL: 以 TWSE API 為主抓取台股日 K（2317、2618），預設回看 3 年；若 TWSE 失敗則以 `yfinance` 作為備援，欄位標準化為 OHLCV。
2. 清洗: 移除格式噪音，對齊交易日，標記可交易日。
3. 特徵工程: 計算 RSI、MACD、Bollinger Bands、SMA(含 10 日)、ATR，另加報酬與波動特徵。
4. 標籤: `target_t1 = 1 (明日收盤 > 今日收盤), 否則 -1`。
5. 防洩漏: 所有特徵一律 `shift(1)`，確保僅使用 t-1 前資訊。
6. 模型: RandomForest + TimeSeriesSplit，並額外提供「前 2 年訓練、剩餘期間測試」的 holdout 評估。
7. 回測: Backtrader 載入模型訊號，硬性交易成本與動態滑價。
8. 監控: 暴露 `/metrics` 觀察延遲、CPU、記憶體與策略指標。

## 金融知識與指標意義
- OHLCV: 市場微觀行為的基本載體。
- RSI: 多空力道，常用於判讀超買超賣。
- MACD: 趨勢與動能強弱。
- Bollinger Bands: 波動收斂/擴張與價格偏離區間。
- SMA: 多尺度趨勢平滑。
- ATR: 波動風險尺度，可作為動態滑價與倉位依據。

## 回測假設（符合需求）
- 手續費: 0.001425
- 證交稅: 0.003
- 滑價: 動態 ATR 比例（上下限保護）

## 評估與解讀
- Accuracy: 整體方向命中率。
- F1-score: 類別不平衡下更穩健。
- Sharpe / Max Drawdown / Win Rate: 報酬-風險-交易品質三者並看。

## 圖表與最終成品
- 回測會輸出每檔 `*_equity_curve.png`（股價正規化與淨值曲線）。
- Notebook 模板：`notebooks/quant_report_template.ipynb`，可匯出 PDF 給面試使用。

## 介面怎麼看
- 報告：`reports/model_report.md`、`reports/backtest_report.md`
- 圖表：`reports/*_equity_curve.png`
- Notebook：`notebooks/quant_report_template.ipynb`
- 監控：`http://localhost:8000/metrics`

## 風險與限制
- 不保證未來報酬或準確率。市場非穩態，模型可能失效。
- 結論需依滾動驗證與持續監控更新。
- 第一階段聚焦結構化資料；新聞等非結構化特徵可於下一階段加入。

## 下一步
- 加入 walk-forward 自動再訓練。
- 加入除權息調整與正式台股交易日曆。
- 比較 RandomForest 與 XGBoost/深度學習基線。
