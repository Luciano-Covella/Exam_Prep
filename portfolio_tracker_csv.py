# Import the necessary libraries (like toolkits for specific tasks)
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from io import StringIO

# Helper functions

def calculate_cagr(start_value: float, end_value: float, periods: float) -> float:
    return (end_value / start_value) ** (1 / periods) - 1


def calculate_max_drawdown(series: pd.Series) -> float:
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def fetch_annual_dividends(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    if dividends.empty:
        return pd.Series(dtype=float)
    idx = dividends.index
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    mask = (idx >= pd.to_datetime(start_date)) & (idx <= pd.to_datetime(end_date))
    filtered = dividends.copy()
    filtered.index = idx
    filtered = filtered.loc[mask]
    return filtered.groupby(filtered.index.year).sum() if not filtered.empty else pd.Series(dtype=float)


def format_market_cap(value: float) -> str:
    """Format market cap into human-readable string with k, Mio., Bn."""
    if pd.isna(value):
        return "N/A"
    if value >= 1e9:
        return f"€{value/1e9:.2f} Bn."
    if value >= 1e6:
        return f"€{value/1e6:.2f} Mio."
    if value >= 1e3:
        return f"€{value/1e3:.2f} k"
    return f"€{value:.2f}"

# App setup
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📊 Portfolio Menu")
    menu = st.radio("Navigation", ["📁 Upload CSV", "📈 Portfolio Overview", "📉 Performance & Risk Analytics"])
    if "last_updated" in st.session_state:
        st.caption(f"Last updated: {st.session_state['last_updated']}")
    if "portfolio_filename" in st.session_state:
        st.caption(f"File: {st.session_state['portfolio_filename']}")

# Session state init
if "portfolio_file" not in st.session_state:
    st.session_state.portfolio_file = None
    st.session_state.portfolio_filename = None

# Upload CSV
if menu == "📁 Upload CSV":
    st.title("Upload Portfolio CSV")
    st.info("Columns: Ticker, Shares, Buy Price, Buy Date")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        st.session_state.portfolio_file = uploaded.read()
        st.session_state.portfolio_filename = uploaded.name
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("File uploaded successfully.")

# Process file
file_content = st.session_state.get('portfolio_file')
if file_content and menu != "📁 Upload CSV":
    df = pd.read_csv(StringIO(file_content.decode('utf-8')))
    df['Buy Date'] = pd.to_datetime(df['Buy Date'])
    today = datetime.today()
    required = ["Ticker","Shares","Buy Price","Buy Date"]
    if not all(c in df.columns for c in required):
        st.error("CSV must contain: Ticker, Shares, Buy Price, Buy Date")
        st.stop()

    # Fetch data and valuations
    names, prices, dividends_map, history_map = {}, [], {}, {}
    for _, row in df.iterrows():
        t, bd = row['Ticker'], row['Buy Date']
        tk = yf.Ticker(t)
        history = tk.history(start=bd, end=today)
        history_map[t] = history
        info = tk.info
        names[t] = info.get('shortName', t)
        prices.append(history['Close'].iloc[-1] if not history.empty else np.nan)
        dividends_map[t] = fetch_annual_dividends(t, bd, today)
        df.loc[df['Ticker'] == t, 'P/E'] = info.get('trailingPE', np.nan)
        df.loc[df['Ticker'] == t, 'Market Cap'] = info.get('marketCap', np.nan)

    # Calculate metrics
    df['Current'] = prices
    df['Value'] = df['Current'] * df['Shares']
    df['Invested'] = df['Shares'] * df['Buy Price']
    df['Abs Perf'] = df['Value'] - df['Invested']
    df['Rel Perf'] = df['Abs Perf'] / df['Invested']
    df['Name'] = df['Ticker'].map(names)
    total_value = df['Value'].sum()
    total_pl = df['Abs Perf'].sum()

            # Portfolio Overview
    if menu == "📈 Portfolio Overview":
        st.title("Portfolio Overview")
        # Only show core columns (remove P/E & Market Cap)
        cols = ['Name', 'Ticker', 'Position Size (€)', 'Absolute Perf (€)', 'Relative Perf (%)']
        # Prepare dataframe for display
        display = df[['Name', 'Ticker', 'Value', 'Abs Perf', 'Rel Perf']].copy()
        display.rename(columns={
            'Value': 'Position Size (€)',
            'Abs Perf': 'Absolute Perf (€)',
            'Rel Perf': 'Relative Perf (%)'
        }, inplace=True)
        display['Relative Perf (%)'] *= 100
        st.dataframe(
            display.style.format({
                'Position Size (€)': '€{:.2f}',
                'Absolute Perf (€)': '€{:.2f}',
                'Relative Perf (%)': '{:.2f}%'
            }),
            use_container_width=True
        )

        # Summary metrics
        st.subheader("Summary")
        c1, c2 = st.columns(2)
        c1.metric("Total Value", f"€{total_value:.2f}")
        c2.metric("Total P/L", f"€{total_pl:.2f}")

        # Allocation pie chart with dynamic font sizing
        st.subheader("Allocation by Value")
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = plt.get_cmap('tab20').colors
        n = len(df)
        fontsize = max(6, 12 - n // 2)
        wedges, texts, autotexts = ax.pie(
            df['Value'],
            labels=df['Ticker'],
            autopct='%1.1f%%',
            startangle=140,
            colors=colors
        )
        for txt in texts + autotexts:
            txt.set_fontsize(fontsize)
        ax.axis('equal')
        st.pyplot(fig)

    # Performance & Risk Analytics
    elif menu == "📉 Performance & Risk Analytics":
        st.title("Performance & Risk Analytics")

        # Per-asset expanders (removed Max Drawdown)
        returns_list = []
        start_date = df['Buy Date'].min().date()
        benchmark = yf.Ticker('^GSPC').history(start=start_date, end=today.date())['Close'].pct_change()
        for t, hist in history_map.items():
            r = hist['Close'].pct_change()
            returns_list.append(r)
            vol = r.std() * np.sqrt(252)
            paired = pd.concat([r, benchmark], axis=1).dropna()
            beta = linregress(paired.iloc[:,1], paired.iloc[:,0])[0] if not paired.empty else np.nan
            pe = df.loc[df['Ticker'] == t, 'P/E'].iloc[0]
            mcap_val = df.loc[df['Ticker'] == t, 'Market Cap'].iloc[0]
            mcap_str = format_market_cap(mcap_val)
            with st.expander(f"{names[t]} ({t})"):
                st.write(f"P/E Ratio: {pe:.2f}")
                st.write(f"Market Cap: {mcap_str}")
                st.write(f"Volatility: {vol:.4f}")
                st.write(f"Beta: {beta:.4f}")

        # Portfolio-level metrics
        if returns_list:
            prt = pd.concat(returns_list, axis=1).mean(axis=1)
            sharpe = prt.mean() / prt.std() * np.sqrt(252)
            downside = prt[prt < 0].std() * np.sqrt(252)
            sortino = prt.mean() / downside if downside else np.nan
            mdd_p = calculate_max_drawdown(prt)
            days = (prt.index[-1] - prt.index[0]).days
            yrs = days / 365.25
            cum_ret = (1 + prt).prod()
            cagr = calculate_cagr(1, cum_ret, yrs)
            tot_ret = (cum_ret - 1) * 100
            vol_p = prt.std() * np.sqrt(252)
            bp = pd.concat([prt, benchmark], axis=1).dropna()
            beta_p = linregress(bp.iloc[:,1], bp.iloc[:,0])[0] if not bp.empty else np.nan

                        # Income Yield: use previous full calendar year dividends
            prev_year = today.year - 1
            total_income = 0
            for ticker, series in dividends_map.items():
                shares = df.loc[df['Ticker'] == ticker, 'Shares'].iloc[0]
                year_div = series.get(prev_year, 0)
                total_income += year_div * shares
            income_yield = (total_income / total_value) * 100 if total_value else 0
