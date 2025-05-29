# Import the necessary libraries (like toolkits for specific tasks)
import streamlit as st  # Streamlit is used to build the interactive web app
import pandas as pd  # Pandas helps load and work with spreadsheet-like data
import yfinance as yf  # yfinance allows downloading stock and crypto prices
import matplotlib.pyplot as plt  # Matplotlib helps create visual charts
import numpy as np  # Numpy handles numerical operations like calculations
from datetime import datetime  # Used to work with dates (like Buy Date)
from scipy.stats import linregress  # Calculates regression (for Beta)
from io import StringIO  # Converts file content into readable text form

# ---------- Helper functions (modular, robust) ----------

def calculate_cagr(start_value: float, end_value: float, periods: float) -> float:
    """Calculate annual growth rate from start to end value over given years."""
    return (end_value / start_value) ** (1 / periods) - 1


def calculate_max_drawdown(series: pd.Series) -> float:
    """Calculate worst drop from peak (risk metric)."""
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def fetch_annual_dividends(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """Fetch and aggregate dividends per year for a given ticker between two dates, handling timezones."""
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    if dividends.empty:
        return pd.Series(dtype=float)
    # Normalize index to naive timestamps
    idx = dividends.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    # Create mask with naive datetimes
    mask = (idx >= pd.to_datetime(start_date)) & (idx <= pd.to_datetime(end_date))
    filtered = dividends.copy()
    filtered.index = idx
    filtered = filtered.loc[mask]
    if filtered.empty:
        return pd.Series(dtype=float)
    return filtered.groupby(filtered.index.year).sum()

# ---------- Streamlit app setup ----------
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# ---------- Sidebar navigation menu ----------
with st.sidebar:
    st.title("📊 Portfolio Menu")
    menu = st.radio(
        "Navigation",
        ["📁 Upload CSV", "📈 Portfolio Overview", "📉 Performance & Risk Analytics"]
    )
    if "last_updated" in st.session_state:
        st.caption(f"Last updated: {st.session_state['last_updated']}")
    if "portfolio_filename" in st.session_state:
        st.caption(f"File: {st.session_state['portfolio_filename']}")

# Initialize session state for file storage
if "portfolio_file" not in st.session_state:
    st.session_state.portfolio_file = None
    st.session_state.portfolio_filename = None

# ---------- Upload CSV ----------
if menu == "📁 Upload CSV":
    st.title("📁 Upload Portfolio CSV")
    st.info("Upload a CSV with columns: Ticker, Shares, Buy Price, Buy Date")
    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded:
        st.session_state.portfolio_file = uploaded.read()
        st.session_state.portfolio_filename = uploaded.name
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("✅ File uploaded successfully. Use the sidebar to continue.")

# ---------- Process uploaded CSV for other pages ----------
file_content = st.session_state.get('portfolio_file')
if file_content and menu != "📁 Upload CSV":
    # Read CSV
    try:
        decoded = StringIO(file_content.decode('utf-8'))
        df = pd.read_csv(decoded)
        if df.empty:
            st.error("❌ Uploaded file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()
    
    # Validate columns
    required = ["Ticker", "Shares", "Buy Price", "Buy Date"]
    if not all(col in df.columns for col in required):
        st.error("❌ CSV must contain: Ticker, Shares, Buy Price, Buy Date")
        st.stop()

    # Parse dates
    df["Buy Date"] = pd.to_datetime(df["Buy Date"])  
    today = datetime.today()

    # Fetch market and dividend data
    names, prices, dividends_map, history_map = {}, [], {}, {}
    for _, row in df.iterrows():
        ticker, buy_date = row['Ticker'], row['Buy Date']
        tk = yf.Ticker(ticker)
        history = tk.history(start=buy_date, end=today)
        history_map[ticker] = history
        names[ticker] = tk.info.get('shortName', ticker) if tk.info else ticker
        dividends_map[ticker] = fetch_annual_dividends(ticker, buy_date, today)
        prices.append(history['Close'].iloc[-1] if not history.empty else np.nan)

    # Compute metrics
    df['Current'] = prices
    df['Value'] = df['Current'] * df['Shares']
    df['Invested'] = df['Shares'] * df['Buy Price']
    df['Abs Perf'] = df['Value'] - df['Invested']
    df['Rel Perf'] = df['Abs Perf'] / df['Invested']
    df['Name'] = df['Ticker'].map(names)

    total_value = df['Value'].sum()
    total_pl = df['Abs Perf'].sum()

    # ---------- Portfolio Overview ----------
    if menu == "📈 Portfolio Overview":
        st.title("📈 Portfolio Overview")

        # Sortable positions list
        st.subheader("Positions")
        sort_option = st.selectbox("Sort by", ["Rel Perf", "Abs Perf", "Value"])
        df_sorted = df.sort_values(by=sort_option, ascending=False)
        for _, r in df_sorted.iterrows():
            col_name, col_metrics = st.columns([3,1])
            with col_name:
                st.markdown(f"**{r['Name']}**  \n<small>{r['Ticker']}</small>", unsafe_allow_html=True)
            with col_metrics:
                st.metric("Size (€)", f"€{r['Value']:.2f}")
                st.metric("Abs (€)", f"€{r['Abs Perf']:.2f}")
                st.metric("Rel (%)", f"{r['Rel Perf']*100:.2f}%")

        # Summary metrics design
        st.subheader("Portfolio Summary")
        m1, m2 = st.columns(2)
        m1.metric("Total Portfolio Value", f"€{total_value:.2f}")
        m2.metric("Total Profit/Loss", f"€{total_pl:.2f}")

        # Allocation pie chart
        st.subheader("Allocation by Value")
        fig1, ax1 = plt.subplots()
        ax1.pie(df['Value'], labels=df['Ticker'], autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Dividends stacked bar
        st.subheader("Received Dividends")
        div_df = pd.DataFrame(dividends_map).fillna(0).sort_index()
        if not div_df.empty:
            fig2, ax2 = plt.subplots()
            div_df.plot(kind='bar', stacked=True, ax=ax2)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Dividends (€)')
            ax2.set_title('Annual Dividends Received')
            st.pyplot(fig2)
        else:
            st.info("No dividend data found for the tickers.")

    # ---------- Performance & Risk Analytics ----------
    elif menu == "📉 Performance & Risk Analytics":
        st.title("📉 Performance & Risk Analytics")
        st.markdown("""
        ### Per Asset Metrics
        - Volatility (Annualized)
        - Max Drawdown
        - Beta vs S&P500

        ### Portfolio Metrics
        - Sharpe Ratio
        - Sortino Ratio
        - Max Drawdown
        - CAGR
        """
        )

        returns_list = []
        start_date = df['Buy Date'].min().date()
        benchmark = yf.Ticker('^GSPC').history(start=start_date, end=today.date())['Close'].pct_change()

        for ticker, hist in history_map.items():
            returns = hist['Close'].pct_change()
            returns_list.append(returns)
            vol = returns.std() * np.sqrt(252)
            mdd = calculate_max_drawdown(returns)
            paired = pd.concat([returns, benchmark], axis=1).dropna()
            beta = linregress(paired.iloc[:,1], paired.iloc[:,0])[0] if not paired.empty else np.nan
            with st.expander(f"📌 {ticker} Metrics"):
                st.write(f"**Volatility:** {vol:.4f}")
                st.write(f"**Max Drawdown:** {mdd:.4f}")
                st.write(f"**Beta vs S&P500:** {beta:.4f}")

        if returns_list:
            port_returns = pd.concat(returns_list, axis=1).mean(axis=1)
            sharpe = port_returns.mean()/port_returns.std()*np.sqrt(252)
            downside = port_returns[port_returns<0].std()*np.sqrt(252)
            sortino = port_returns.mean()/downside if downside else np.nan
            port_mdd = calculate_max_drawdown(port_returns)
            days = (port_returns.index[-1]-port_returns.index[0]).days
            years = days/365.25
            cum_ret = (1+port_returns).prod()
            cagr = calculate_cagr(1, cum_ret, years)

            st.subheader("Portfolio Summary")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Sharpe", round(sharpe,3))
            p2.metric("Sortino", round(sortino,3))
            p3.metric("Max Drawdown", round(port_mdd,3))
            p4.metric("CAGR", f"{cagr*100:.2f}%")

            st.subheader("Cumulative Return")
            cum = (1+port_returns).cumprod()
            fig3, ax3 = plt.subplots()
            ax3.plot(cum.index, cum.values)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Cumulative Return')
            st.pyplot(fig3)
