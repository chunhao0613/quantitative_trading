import sys
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots

# 確保可以匯入專案的 src
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.etl.fetch_twse_tpex import fetch_market_data
from src.etl.clean_align import clean_and_align
from src.features.technical_indicators import add_indicators
from src.features.build_dataset import build_dataset
from src.models.train_eval import evaluate_symbol
from src.backtest.run_backtest import run_single_backtest

# 設定 Streamlit 頁面
st.set_page_config(page_title="量化交易展示平台", page_icon="📈", layout="wide")
st.title("台灣股市量化特徵觀測站")

DB_PATH = ROOT / "market_data.db"

# 建立快取連線
@st.cache_resource
def get_db_connection():
    # check_same_thread=False 讓 Streamlit 多執行緒可以共用
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def get_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_db_connection()
    
    # 建立表格（若不存在）
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_features (
            date TEXT,
            symbol TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            log_ret_1 REAL, log_ret_5 REAL, 
            dist_sma_5 REAL, dist_sma_10 REAL, dist_sma_20 REAL, dist_sma_60 REAL,
            macd_hist_z REAL, rsi_14 REAL, rsi_14_z REAL, 
            atr_14 REAL, atr_ratio REAL, vol_5 REAL, vol_20 REAL
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_metrics (
            symbol TEXT,
            date TEXT,
            ic REAL,
            rank_ic REAL,
            holdout_ic REAL,
            sharpe REAL,
            total_return REAL,
            max_drawdown REAL,
            win_rate REAL
        )
    ''')
    
    # 建立「複合索引 (Composite Index)」，這是解決資料變多變慢的關鍵！
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_symbol_date 
        ON daily_features (symbol, date)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_metrics_symbol 
        ON daily_metrics (symbol)
    ''')
    
    df_sql = pd.DataFrame()
    # 嘗試從 SQL 撈取該股票資料
    try:
        query = f"SELECT * FROM daily_features WHERE symbol = '{symbol}' ORDER BY date"
        df_sql = pd.read_sql(query, conn)
    except Exception as e:
        st.warning(f"資料庫讀取異常: {e}")

    needs_fetch = False
    
    if df_sql.empty:
        st.info(f"🔴 [SQL Cache Miss] 資料庫沒有 {symbol} 的資料，開始從 TWSE API 抓取...")
        needs_fetch = True
    else:
        # 檢查資料庫最新的一筆資料日期是否接近目標結束日期 (容許 3 天誤差，包含週末)
        df_sql['date'] = pd.to_datetime(df_sql['date'])
        max_date = df_sql['date'].max()
        end_dt = pd.to_datetime(end_date)
        
        # 檢查是否具備指標紀錄
        metrics_exist = False
        try:
            m_count = pd.read_sql(f"SELECT COUNT(*) as c FROM daily_metrics WHERE symbol = '{symbol}'", conn).iloc[0]['c']
            if m_count > 0:
                metrics_exist = True
        except Exception:
            pass
        
        if (end_dt - max_date).days > 3 or not metrics_exist:
            st.info(f"🟡 [SQL Update Needed] 資料庫資料已舊或缺乏指標紀錄，重新跑分析管線...")
            needs_fetch = True
            
            # 清除舊資料，準備覆蓋
            conn.execute(f"DELETE FROM daily_features WHERE symbol = '{symbol}'")
            conn.execute(f"DELETE FROM daily_metrics WHERE symbol = '{symbol}'")
            conn.commit()
        else:
            st.success(f"🟢 [SQL Cache Hit] 成功從本地資料庫快速讀取 {symbol} 資料！")

    if needs_fetch:
        with st.spinner('正在從交易所獲取原始資料並計算技術指標...'):
            try:
                # 1. 抓取原始資料
                raw_df = fetch_market_data(
                    stock_no=symbol, 
                    start=start_date, 
                    end=end_date,
                    data_source="twse",
                    allow_synthetic=False,
                    force_synthetic=False
                )
                
                # 2. 清洗與對齊，並寫入 Parquet 供管線使用
                clean_df = clean_and_align(raw_df, forward_fill=True)
                clean_path = ROOT / f"data/processed/{symbol}_clean.parquet"
                clean_path.parent.mkdir(parents=True, exist_ok=True)
                clean_df.to_parquet(clean_path, index=False)
                
                # 3. 計算技術特徵，並寫入 Parquet
                feat_df = add_indicators(clean_df)
                feat_path = ROOT / f"data/processed/{symbol}_features.parquet"
                feat_df.to_parquet(feat_path, index=False)
                
                # 4. 建立訓練資料集
                ds_df = build_dataset(feat_df)
                ds_path = ROOT / f"data/processed/{symbol}_dataset.parquet"
                ds_df.to_parquet(ds_path, index=False)
                
                # 5. 執行模型訓練與評估 (產出 IC, Rank IC)
                report_dir = ROOT / "reports"
                model_dir = ROOT / "artifacts"
                report_dir.mkdir(parents=True, exist_ok=True)
                model_dir.mkdir(parents=True, exist_ok=True)
                
                metrics = evaluate_symbol(ds_path, report_dir, model_dir)
                signal_path = report_dir / f"{symbol}_signals.parquet"
                
                # 6. 執行回測 (產出 Sharpe, Max Drawdown)
                bt_metrics = run_single_backtest(signal_path, clean_path)
                
                # 確保 date 轉換回字串才能存入 SQLite
                feat_df['date'] = feat_df['date'].astype(str)
                
                # 選取要存入 SQL 的欄位
                cols_to_save = [
                    'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                    'log_ret_1', 'log_ret_5', 
                    'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_60',
                    'macd_hist_z', 'rsi_14', 'rsi_14_z', 
                    'atr_14', 'atr_ratio', 'vol_5', 'vol_20'
                ]
                # 有些欄位可能在指標計算初期是 NaN，這裡不用全清
                save_df = feat_df[cols_to_save].copy()
                
                # 7. 存入 SQL (特徵資料)
                save_df.to_sql('daily_features', conn, if_exists='append', index=False)
                
                # 8. 存入 SQL (指標資料)
                metrics_df = pd.DataFrame([{
                    'symbol': symbol,
                    'date': end_date,
                    'ic': metrics.get('ic', 0.0),
                    'rank_ic': metrics.get('rank_ic', 0.0),
                    'holdout_ic': metrics.get('holdout', {}).get('ic', 0.0),
                    'sharpe': bt_metrics.get('sharpe', 0.0),
                    'total_return': bt_metrics.get('total_return', 0.0),
                    'max_drawdown': bt_metrics.get('max_drawdown', 0.0),
                    'win_rate': bt_metrics.get('win_rate', 0.0)
                }])
                metrics_df.to_sql('daily_metrics', conn, if_exists='append', index=False)
                
                st.success(f"💾 {symbol} 資料抓取、模型訓練與回測完成，已存入 SQL 快取。")
                
                df_sql = save_df
                df_sql['date'] = pd.to_datetime(df_sql['date'])
                
            except Exception as e:
                st.error(f"抓取失敗: {e}")
                return pd.DataFrame()
                
    return df_sql

# UI 區塊
with st.sidebar:
    st.header("參數設定")
    symbol_input = st.text_input("股票代號", value="2330")
    
    # 預設抓取過去一年到今天
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)
    
    start_d = st.date_input("開始日期", value=one_year_ago)
    end_d = st.date_input("結束日期", value=today)
    
    fetch_btn = st.button("獲取資料 / 顯示圖表", type="primary")

if fetch_btn:
    df = get_stock_data(
        symbol=symbol_input,
        start_date=start_d.strftime("%Y-%m-%d"),
        end_date=end_d.strftime("%Y-%m-%d")
    )
    
    if not df.empty:
        # --- 策略穩定性指標區塊 (KPI 儀表板) ---
        conn = get_db_connection()
        metrics_sql = pd.read_sql(f"SELECT * FROM daily_metrics WHERE symbol = '{symbol_input}' ORDER BY date DESC LIMIT 1", conn)
        
        st.markdown("---")
        st.subheader("💡 策略績效與穩定性指標 (Backtest & Model Stability)")
        if not metrics_sql.empty:
            m = metrics_sql.iloc[0]
            
            # 使用 Streamlit columns 打造舒適的排版
            c1, c2, c3, c4, c5 = st.columns(5)
            
            # 定義顏色與箭頭邏輯
            ic_val = m['ic']
            ic_color = "normal" if ic_val > 0.02 else "inverse"
            c1.metric("Information Coefficient (IC)", f"{ic_val:.4f}", help="模型在全區間的預測方向準確度，>0.02 具微弱訊號", delta="整體 IC", delta_color=ic_color)
            
            h_ic_val = m['holdout_ic']
            h_ic_color = "normal" if h_ic_val > 0.02 else "inverse"
            c2.metric("Holdout IC (盲測)", f"{h_ic_val:.4f}", help="模型在20%測試資料預測，>0.02有略微的預測能力、>0.05有不錯的預測能力、<0.02代表有嚴重的過擬傷", delta="樣本外 IC", delta_color=h_ic_color)
            
            sharpe_val = m['sharpe']
            sharpe_color = "normal" if sharpe_val > 1.0 else "off"
            c3.metric("Sharpe Ratio", f"{sharpe_val:.2f}", help="風險調整後報酬，大於 1 為佳", delta="Sharpe", delta_color=sharpe_color)
            
            ret_val = m['total_return'] * 100
            ret_color = "normal" if ret_val > 0 else "inverse"
            c4.metric("Total Return", f"{ret_val:.2f}%", help="回測區間總報酬率", delta="Return", delta_color=ret_color)
            
            mdd_val = m['max_drawdown'] * 100
            c5.metric("Max Drawdown", f"{mdd_val:.2f}%", help="區間最大虧損幅度，越接近 0 越好", delta="Risk", delta_color="inverse")
            
        else:
            st.warning("目前尚無此檔股票的回測指標記錄。")
            
        st.markdown("---")
        st.subheader(f"📊 {symbol_input} 價格與量化特徵分析")
        
        # 使用 Plotly 畫圖 (K線 + MACD Z-Score + RSI)
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2]
        )

        # 1. K線圖
        fig.add_trace(
            plotly_go.Candlestick(
                x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                name="K線"
            ),
            row=1, col=1
        )
        
        # 加上 SMA 20
        sma_20 = df['close'] / (1 + df['dist_sma_20']) # 從距離反推 SMA
        fig.add_trace(
            plotly_go.Scatter(x=df['date'], y=sma_20, mode='lines', name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )

        # 2. MACD Z-Score
        fig.add_trace(
            plotly_go.Bar(
                x=df['date'], y=df['macd_hist_z'], 
                name="MACD Z-Score",
                marker_color=['green' if val >= 0 else 'red' for val in df['macd_hist_z']]
            ),
            row=2, col=1
        )

        # 3. RSI
        fig.add_trace(
            plotly_go.Scatter(x=df['date'], y=df['rsi_14'], mode='lines', name='RSI 14', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示資料表
        st.subheader("原始資料與衍生特徵 (Data & Features)")
        st.dataframe(df.sort_values("date", ascending=False).head(100), use_container_width=True)
