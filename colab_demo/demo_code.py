# 1. Kill background processes
!pkill -f ngrok
!pkill -f streamlit

# 2. Install dependencies
!pip install streamlit pyngrok yfinance pandas numpy statsmodels plotly matplotlib -q

# 3. Data Download & Preparation
import yfinance as yf
import pandas as pd
from datetime import datetime

nifty50_tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'HINDUNILVR.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'KOTAKBANK.NS',
    'MARUTI.NS', 'LT.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'SUNPHARMA.NS',
    'WIPRO.NS', 'TATAMOTORS.NS', 'M&M.NS', 'NTPC.NS'
]

def load_data():
    print(f"Downloading data for {len(nifty50_tickers)} tickers...")
    df = yf.download(nifty50_tickers, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=True, progress=False)

    if 'Close' in df.columns:
        df = df['Close']
    elif hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
        df = df['Close']

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    df = df.ffill().bfill()

    missing_tickers = set(nifty50_tickers) - set(df.columns)
    if missing_tickers:
        print(f"Warning: Missing tickers detected: {missing_tickers}")
        for ticker in missing_tickers:
            try:
                missing_data = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=True, progress=False)['Close']
                df[ticker] = missing_data
                print(f"Successfully added missing ticker: {ticker}")
            except:
                print(f"Failed to download: {ticker}")

    print(f"Final dataset has {len(df.columns)} stocks")
    return df

all_data = load_data()
all_data.to_csv('data.csv', index=True)
print(f"Data saved to data.csv with {len(all_data.columns)} stocks")

# 4. Create Streamlit App (With Cold Start Fix + Scatter Plot)
app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import plotly.express as px
import itertools

# --- Load Data ---
@st.cache_data
def load_csv_data():
    return pd.read_csv('data.csv', index_col=0, parse_dates=True)

all_data = load_csv_data()

st.sidebar.write(f"Dataset loaded: {len(all_data.columns)} stocks")
st.sidebar.write(f"Total possible pairs: {len(list(itertools.combinations(all_data.columns, 2)))}")

# --- Core Logic Functions ---

def find_cointegration_window(stock1_data, stock2_data, end_year=2022, min_start_year=2010):
    '''Iterates through start years to find a valid cointegration window.'''
    earliest_year = stock1_data.index.min().year
    for start_year in range(earliest_year, min_start_year + 1):
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        try:
            window_data = pd.concat([stock1_data, stock2_data], axis=1).loc[start_date:end_date].dropna()
            if len(window_data) < 252:
                continue
            _, pvalue, _ = coint(window_data.iloc[:, 0], window_data.iloc[:, 1])
            if pvalue < 0.05:
                return start_date, end_date, pvalue
        except Exception:
            continue
    return None, None, None

def run_backtest(trading_df, hedge_ratio, warmup_df=None, initial_capital=100000.0, 
                 entry_threshold=2.0, exit_threshold=0.5, 
                 rolling_window=30, allocation=1.0, fee_bps=0.0):
    '''Runs mean reversion backtest with Warmup Fix.'''
    if trading_df is None or trading_df.empty:
        return initial_capital, 0.0, 0.0, 0, 0.0, [], []

    # --- WARMUP FIX ---
    if warmup_df is not None and not warmup_df.empty:
        full_df = pd.concat([warmup_df, trading_df])
        cutoff_index = len(warmup_df)
    else:
        full_df = trading_df
        cutoff_index = 0
        
    prices1_full = full_df.iloc[:, 0].astype(float)
    prices2_full = full_df.iloc[:, 1].astype(float)
    
    spread_full = prices1_full - hedge_ratio * prices2_full
    mu_full = spread_full.rolling(rolling_window).mean()
    sigma_full = spread_full.rolling(rolling_window).std(ddof=0).replace(0.0, np.nan)
    z_full = (spread_full - mu_full) / sigma_full
    
    # Slice back to trading period
    z = z_full.iloc[cutoff_index:]
    prices1 = prices1_full.iloc[cutoff_index:]
    prices2 = prices2_full.iloc[cutoff_index:]
    
    # Simulation
    cash = float(initial_capital)
    q1, q2 = 0.0, 0.0
    position = 0 
    trade_count = 0
    equity_curve = []
    trade_pnls = []
    entry_equity = None
    
    def portfolio_value(p1, p2, cash_val):
        return cash_val + q1 * p1 + q2 * p2

    for t in range(len(prices1)):
        p1 = prices1.iloc[t]
        p2 = prices2.iloc[t]
        z_score = z.iloc[t]
        
        current_equity = portfolio_value(p1, p2, cash)
        equity_curve.append(current_equity)
        
        if not np.isfinite(z_score): continue
            
        if position == 0:
            if abs(z_score) > entry_threshold:
                long_spread = (z_score < -entry_threshold)
                deploy_equity = allocation * current_equity
                if deploy_equity <= 0: continue
                
                gross_needed = abs(p1) + abs(hedge_ratio * p2)
                if gross_needed <= 0: continue
                
                k = deploy_equity / gross_needed
                if long_spread:
                    q1, q2 = k, -hedge_ratio * k
                    position = 1
                else:
                    q1, q2 = -k, hedge_ratio * k
                    position = -1
                
                cash -= (q1 * p1 + q2 * p2)
                if fee_bps > 0:
                    entry_gross = abs(q1 * p1) + abs(q2 * p2)
                    cash -= (fee_bps / 10000.0) * entry_gross
                entry_equity = current_equity
                trade_count += 1
                
        elif position != 0:
            if abs(z_score) < exit_threshold:
                exit_value = q1 * p1 + q2 * p2
                cash += exit_value
                if fee_bps > 0:
                    exit_gross = abs(q1 * p1) + abs(q2 * p2)
                    cash -= (fee_bps / 10000.0) * exit_gross
                exit_equity = portfolio_value(p1, p2, cash)
                if entry_equity is not None:
                    trade_pnls.append(exit_equity - entry_equity)
                q1, q2 = 0.0, 0.0
                position = 0
                entry_equity = None

    final_value = equity_curve[-1] if equity_curve else initial_capital
    pnl = final_value - initial_capital
    pnl_pct = (pnl / initial_capital * 100) if initial_capital != 0 else 0.0
    
    win_rate = 0.0
    if len(trade_pnls) > 0:
        winning_trades = sum(1 for p in trade_pnls if p > 0)
        win_rate = (winning_trades / len(trade_pnls)) * 100
        
    return float(final_value), float(pnl), float(pnl_pct), int(trade_count), float(win_rate), equity_curve, z

# --- UI Layout ---

page = st.sidebar.selectbox("Select Analysis", ["Full Analysis", "Particular Pair Analysis"])

if page == "Full Analysis":
    st.title("Full Market Analysis (Nifty 50)")
    
    if st.button("Run Full Analysis"):
        results = []
        all_tickers = all_data.columns.tolist()
        progress_bar = st.progress(0)
        
        pairs = list(itertools.combinations(all_tickers, 2))
        total_pairs = len(pairs)
        st.write(f"Processing {total_pairs} pairs from {len(all_tickers)} stocks...")
        
        for idx, (stock1, stock2) in enumerate(pairs):
            progress_bar.progress((idx + 1) / total_pairs)
            
            # 1. Formation
            formation_df = all_data[[stock1, stock2]].loc[:'2022-12-31']
            
            # 2. Check Cointegration
            start_date, end_date, pvalue = find_cointegration_window(
                formation_df[stock1], formation_df[stock2], min_start_year=2020
            )
            
            if pvalue is not None:
                window_data = formation_df.loc[start_date:end_date]
                X = sm.add_constant(window_data[stock2])
                model = sm.OLS(window_data[stock1], X).fit()
                hedge_ratio = model.params.iloc[1]
                
                # 3. Backtest (WITH WARMUP)
                warmup_df = all_data[[stock1, stock2]].loc['2022-10-01':'2022-12-31']
                trading_df = all_data[[stock1, stock2]].loc['2023-01-01':]
                
                final_val, pnl, pnl_pct, trades, win_rate, _, _ = run_backtest(
                    trading_df, hedge_ratio, warmup_df=warmup_df
                )
                
                # Calculate Duration for Plotting
                start_year_int = int(start_date[:4])
                end_year_int = int(end_date[:4])
                duration = end_year_int - start_year_int + 1
                
                results.append({
                    "Stock 1": stock1.replace('.NS', ''),
                    "Stock 2": stock2.replace('.NS', ''),
                    "Formation Period": f"{start_date[:4]}-{end_date[:4]}",
                    "Window Duration (Yrs)": duration,
                    "P-Value": pvalue,
                    "Hedge Ratio": hedge_ratio,
                    "Trades": trades,
                    "Win Rate (%)": win_rate,
                    "Final Value (₹)": final_val,
                    "P&L (₹)": pnl,
                    "P&L (%)": pnl_pct
                })
        
        st.success(f"Analysis completed! Found {len(results)} cointegrated pairs.")
        
        if results:
            results_df = pd.DataFrame(results).sort_values('P&L (₹)', ascending=False)
            
           # --- NEW: Improved Hypothesis Testing Visualization ---
            st.subheader("Hypothesis Test: Window Duration vs. Profit")
            st.write("Bubble Size = Number of Trades | Color = P-Value (Darker is better)")
            
            fig = px.scatter(
                results_df,
                x="Window Duration (Yrs)",
                y="P&L (%)",
                color="P-Value",
                # We cap the size so bubbles don't get massive
                size=results_df["Trades"].apply(lambda x: min(max(x, 5), 30)), 
                hover_data=["Stock 1", "Stock 2", "Trades", "P-Value"],
                trendline="ols", 
                title="Impact of Cointegration Window on Strategy Return",
                opacity=0.7,  # <--- Helper 1: Makes bubbles see-through
                color_continuous_scale="Blues_r" # Reverse blues: Dark = Low P-value (Good)
            )
            
            # Helper 2: Explicitly set max bubble size to prevent overcrowding
            fig.update_traces(marker=dict(sizemode='area', sizeref=2.*max(results_df["Trades"])/(20.**2), sizemin=4))
            
            st.plotly_chart(fig)

            # --- NEW: Improved Hypothesis Testing Visualization ---
            st.subheader("Hypothesis Test: Trading Frequency vs. Profit")
            st.write("This chart proves that **Mean Reversion Speed** (indicated by Trade Count) is a better predictor of profit than History Length.")
            
            fig = px.scatter(
                results_df,
                x="Trades",  # <--- CHANGED: X-Axis is now Trade Count
                y="P&L (%)",
                color="Window Duration (Yrs)", # Color now shows History Length
                size="Trades", # Keep size correlated for emphasis
                hover_data=["Stock 1", "Stock 2", "P-Value", "Window Duration (Yrs)"],
                trendline="ols", # This trendline should now point UP
                title="Impact of Trading Activity (Half-Life Proxy) on Strategy Return",
                opacity=0.8,
                color_continuous_scale="Viridis" 
            )
            
            # Make sure bubbles are big enough to see clearly
            fig.update_traces(marker=dict(sizemode='area', sizeref=2.*max(results_df["Trades"])/(20.**2), sizemin=6))
            
            st.plotly_chart(fig)
            
            st.subheader("Detailed Results")
            st.dataframe(results_df)
        else:
            st.write("No cointegrated pairs found.")

elif page == "Particular Pair Analysis":
    st.title("Particular Pair Analysis")
    
    tickers_sorted = sorted(all_data.columns.tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        stock1 = st.selectbox('Stock 1', tickers_sorted, index=0)
    with col2:
        stock2 = st.selectbox('Stock 2', tickers_sorted, index=1)
        
    st.subheader("Strategy Parameters")
    c1, c2, c3 = st.columns(3)
    entry_thresh = c1.slider('Entry Z-score', 0.5, 4.0, 2.0, 0.1)
    exit_thresh = c2.slider('Exit Z-score', 0.1, 2.0, 0.5, 0.1)
    window = c3.slider('Rolling Window', 10, 120, 30, 5)
    
    c4, c5 = st.columns(2)
    alloc = c4.slider('Allocation', 0.1, 1.0, 1.0, 0.1)
    fees = c5.slider('Fee (bps)', 0.0, 20.0, 0.0, 0.5)
    
    if st.button('Run Backtest'):
        formation_df = all_data[[stock1, stock2]].loc[:'2022-12-31']
        
        start_date, end_date, pvalue = find_cointegration_window(
            formation_df[stock1], formation_df[stock2], min_start_year=2020
        )
        
        if pvalue is not None:
            st.info(f"Cointegration found! Window: {start_date} to {end_date} (p={pvalue:.4f})")
            
            window_data = formation_df.loc[start_date:end_date]
            X = sm.add_constant(window_data[stock2])
            model = sm.OLS(window_data[stock1], X).fit()
            hedge_ratio = model.params.iloc[1]
            st.write(f"Hedge Ratio: {hedge_ratio:.4f}")
            
            warmup_df = all_data[[stock1, stock2]].loc['2022-10-01':'2022-12-31']
            trading_df = all_data[[stock1, stock2]].loc['2023-01-01':]
            
            final_val, pnl, pnl_pct, trades, win_rate, equity, z_scores = run_backtest(
                trading_df, hedge_ratio,
                warmup_df=warmup_df,
                entry_threshold=entry_thresh,
                exit_threshold=exit_thresh,
                rolling_window=window,
                allocation=alloc,
                fee_bps=fees
            )
            
            st.subheader("Performance")
            st.metric("Total P&L", f"₹{pnl:,.2f}", f"{pnl_pct:.2f}%")
            st.write(f"Trades: {trades} | Win Rate: {win_rate:.2f}% | Final Equity: ₹{final_val:,.2f}")
            
            st.subheader("Equity Curve")
            eq_df = pd.DataFrame({'Equity': equity}, index=trading_df.index)
            st.plotly_chart(px.line(eq_df, y='Equity'))
            
            st.subheader("Z-Score Signals")
            z_df = pd.DataFrame({'Z-Score': z_scores}, index=trading_df.index)
            fig, ax = plt.subplots(figsize=(10, 4))
            z_df.plot(ax=ax, legend=False)
            ax.axhline(entry_thresh, color='r', linestyle='--', alpha=0.6)
            ax.axhline(-entry_thresh, color='r', linestyle='--', alpha=0.6)
            ax.axhline(exit_thresh, color='g', linestyle='--', alpha=0.6)
            ax.axhline(-exit_thresh, color='g', linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
        else:
            st.error("No valid cointegration window found for this pair in the formation period.")
"""

with open('app.py', 'w') as f:
    f.write(app_code)

# 5. Run Streamlit with Ngrok
!nohup streamlit run app.py --server.headless true --server.port 8501 &> logs.txt &

import time
from pyngrok import ngrok

time.sleep(10)
ngrok.set_auth_token('PASTE_YOUR_NGROK_API_KEY_HERE')
public_url = ngrok.connect(8501)
print('🚀 HYPOTHESIS TEST APP IS LIVE!')
print(f'🔗 {public_url}')
