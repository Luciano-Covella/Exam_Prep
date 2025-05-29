# Import the necessary libraries (like toolkits for specific tasks)
import streamlit as st  # Streamlit is used to build the interactive web app
import pandas as pd  # Pandas helps load and work with spreadsheet-like data
import yfinance as yf  # yfinance allows downloading stock and crypto prices
import matplotlib.pyplot as plt  # Matplotlib helps create visual charts
import numpy as np  # Numpy handles numerical operations like calculations
from datetime import datetime  # Used to work with dates (like Buy Date)
from scipy.stats import linregress  # Calculates regression (for Beta)
from io import StringIO  # Converts file content into readable text form

# ---------- Helper functions (modular, bug-free) ----------

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
    """Fetch and aggregate dividends per year for a given ticker between two dates."""
    stock = yf.Ticker(ticker)
    dividends = stock.dividends.loc[start_date:end_date]
    return dividends.groupby(dividends.index.year).sum() if not dividends.empty else pd.Series(dtype=float)

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

# ---------- Initialize session state ----------
if "portfolio_file" not in st.session_state:
    st.session_state.portfolio_file = None
    st.session_state.portfolio_filename = None

# ---------- File upload section ----------
if menu == "📁 Upload CSV":
    st.title("📁 Upload Portfolio CSV")
    st.info("Upload a CSV with columns: Ticker, Shares, Buy Price, Buy Date")
    uploaded = st.file_uploader("Upload CSV File", type=["csv"] )
    if uploaded:
        st.session_state.portfolio_file = uploaded.read()
        st.session_state.portfolio_filename = uploaded.name
        st.success("✅ File uploaded successfully. Use the sidebar to continue.")
        st.session_state['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- Read and parse file content if available ----------
file_content = st.session_state.get('portfolio_file', None)
if file_content:
    try:
        decoded = StringIO(file_content.decode('utf-8'))
        df = pd.read_csv(decoded)
        if df.empty:
            st.error("❌ The uploaded file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {str(e)}")
        st.stop()

    required_cols = ["Ticker", "Shares", "Buy Price", "Buy Date"]
    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV must contain: Ticker, Shares, Buy Price, Buy Date")
        st.stop()

    df["Buy Date"] = pd.to_datetime(df["Buy Date"])
    current_prices, names_data, dividends_data, historical_data = [], {}, {}, {}

    st.write("⏳ Fetching data...")
    for _, row in df.iterrows():
        ticker = row['Ticker']
        buy_date = row['Buy Date']
        today = datetime.today()

        stock = yf.Ticker(ticker)
        # Price history
        history = stock.history(start=buy_date, end=today)
        historical_data[ticker] = history
        # Company name
        try:
            names_data[ticker] = stock.info.get('shortName', ticker)
        except Exception:
            names_data[ticker] = ticker
        # Annual dividends
        dividends_data[ticker] = fetch_annual_dividends(ticker, buy_date, today)
        # Current price
        try:
            current_prices.append(history['Close'].iloc[-1])
        except Exception:
            current_prices.append(None)

    # Core portfolio calculations
    df['Current Price'] = current_prices
    df['Value'] = df['Current Price'] * df['Shares']
    df['Profit/Loss'] = (df['Current Price'] - df['Buy Price']) * df['Shares']
    df['Name'] = df['Ticker'].map(names_data)
    df['Invested'] = df['Shares'] * df['Buy Price']
    df['Absolute Performance'] = df['Profit/Loss']
    df['Relative Performance'] = df['Profit/Loss'] / df['Invested']
    df['Position Size'] = df['Value']

    total_value = df['Value'].sum()
    total_gain = df['Profit/Loss'].sum()

    # ---------- Portfolio Overview Section ----------
    if menu == "📈 Portfolio Overview":
        st.title("📈 Portfolio Overview")
        # Sortable Positions list
        st.subheader("Positions")
        sort_option = st.selectbox(
            "Sort by:",
            ["Relative Performance", "Absolute Performance", "Position Size"]
        )
        ascending = False
        df_sorted = df.sort_values(by=sort_option, ascending=ascending)

        for _, r in df_sorted.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{r['Name']}**  
<small>{r['Ticker']}</small>", unsafe_allow_html=True)
            with col2:
                st.metric(label="Size (€)", value=f"€{r['Position Size']:.2f}")
                st.metric(label="Abs Perf (€)", value=f"€{r['Absolute Performance']:.2f}")
                st.metric(label="Rel Perf (%)", value=f"{r['Relative Performance']*100:.2f}%")

        # Overall metrics
        st.subheader("💰 Portfolio Summary")
        m1, m2 = st.columns(2)
        m1.metric("Total Portfolio Value", f"€{total_value:.2f}")
        m2.metric("Total Profit/Loss", f"€{total_gain:.2f}")

        # Allocation by Value
        st.subheader("📊 Allocation by Value")
        fig1, ax1 = plt.subplots()
        ax1.pie(df['Value'], labels=df['Ticker'], autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Received Dividends Chart
        st.subheader("📊 Received Dividends")
        annual_divs = pd.DataFrame(dividends_data).fillna(0).sort_index()
        if not annual_divs.empty:
            fig2, ax2 = plt.subplots()
            annual_divs.plot(kind='bar', stacked=True, ax=ax2)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Dividends (€)')
            ax2.set_title('Annual Dividends Received')
            st.pyplot(fig2)
        else:
            st.info("No dividend data available.")

    # ---------- Risk and Performance Section ----------
    elif menu == "📉 Performance & Risk Analytics":
        st.title("📉 Performance and Risk Analytics")  # Page heading

        st.markdown("""
        This section shows:
        - Per Asset: Volatility, Max Drawdown, Beta vs S&P500
        - Whole Portfolio: Sharpe Ratio, Sortino Ratio, Max Drawdown, CAGR
        """)

        returns = []  # List to store each asset's returns
        start_date = df["Buy Date"].min().date()  # Get earliest Buy Date
        benchmark = yf.Ticker("^GSPC").history(start=start_date, end=datetime.today().date())["Close"].pct_change()  # S&P500 % changes

        for ticker, history in historical_data.items():
            history["Return"] = history["Close"].pct_change()  # Daily returns for the asset
            returns.append(history["Return"])  # Add to list

            volatility = history["Return"].std() * np.sqrt(252)  # Annualized volatility
            max_dd = calculate_max_drawdown(history["Return"])  # Worst loss
            aligned = pd.concat([history["Return"], benchmark], axis=1).dropna()  # Match with S&P500

            if not aligned.empty:
                slope, *_ = linregress(aligned.iloc[:, 1], aligned.iloc[:, 0])  # Beta value
            else:
                slope = np.nan  # No data for beta

            with st.expander(f"📌 {ticker} Metrics"):  # Expandable view per asset
                st.write("**Volatility (Annualized):**", round(volatility, 4))
                st.write("**Max Drawdown:**", round(max_dd, 4))
                st.write("**Beta vs S&P500:**", round(slope, 4))

        # ---------- Portfolio-wide stats ----------
        if returns:
            combined_returns = pd.concat(returns, axis=1).mean(axis=1)  # Average daily return of all assets

            sharpe = combined_returns.mean() / combined_returns.std() * np.sqrt(252)  # Risk-adjusted return
            downside = combined_returns[combined_returns < 0].std() * np.sqrt(252)  # Negative returns deviation
            sortino = combined_returns.mean() / downside if downside else np.nan  # Sortino ratio
            max_dd = calculate_max_drawdown(combined_returns)  # Portfolio max drawdown

            days = (combined_returns.index[-1] - combined_returns.index[0]).days  # Total number of days
            years = days / 365.25  # Convert to years
            cumulative_return = (1 + combined_returns).prod()  # Total return
            cagr = calculate_cagr(1, cumulative_return, years)  # Compound annual growth rate

            st.subheader("📦 Portfolio Summary")  # Metrics section
            st.metric("Sharpe Ratio", round(sharpe, 3))  # Display Sharpe
            st.metric("Sortino Ratio", round(sortino, 3))  # Display Sortino
            st.metric("Max Drawdown", round(max_dd, 3))  # Display max drop
            st.metric("CAGR", f"{round(cagr * 100, 2)}%")  # Display CAGR

            st.subheader("📈 Cumulative Return")  # Performance graph
            cumulative = (1 + combined_returns).cumprod()  # Build return curve
            fig2, ax2 = plt.subplots()
            ax2.plot(cumulative.index, cumulative.values)  # Draw curve
            ax2.set_title("Cumulative Portfolio Return")  # Add title
            ax2.set_xlabel("Date")  # X-axis
            ax2.set_ylabel("Cumulative Return")  # Y-axis
            st.pyplot(fig2)  # Show chart



