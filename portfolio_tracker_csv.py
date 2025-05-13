import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from io import StringIO

# Configure Streamlit page
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# Sidebar Navigation (Burger-Menü)
with st.sidebar:
    st.title("📊 Portfolio Menu")
    menu = st.radio(
        "Navigation",
        ["📁 Upload CSV", "📈 Portfolio Overview", "📉 Performance & Risk Analytics"]
    )

# Session-State vorbereiten
if "portfolio_file" not in st.session_state:
    st.session_state.portfolio_file = None
    st.session_state.portfolio_filename = None

# ============ CSV Upload ============
if menu == "📁 Upload CSV":
    st.title("📁 Upload Portfolio CSV")
    st.info("Upload a CSV with columns: Ticker, Shares, Buy Price, Buy Date")

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded:
        # Datei-Inhalt lesen und speichern (als Bytes)
        st.session_state.portfolio_file = uploaded.read()
        st.session_state.portfolio_filename = uploaded.name
        st.success("✅ File uploaded successfully. Use the sidebar to continue.")

# ============ Datei einlesen ============
file_content = st.session_state.get("portfolio_file", None)

if file_content:
    try:
        decoded = StringIO(file_content.decode("utf-8"))
        df = pd.read_csv(decoded)

        if df.empty:
            st.error("❌ The uploaded file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {str(e)}")
        st.stop()

    # Spalten prüfen
    required_cols = ["Ticker", "Shares", "Buy Price", "Buy Date"]
    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV must contain: Ticker, Shares, Buy Price, Buy Date")
        st.stop()

    # Vorbereitung
    df["Buy Date"] = pd.to_datetime(df["Buy Date"])
    current_prices = []
    historical_data = {}

    st.write("⏳ Fetching historical and current prices...")

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        buy_date = row["Buy Date"]
        today = datetime.today()

        stock = yf.Ticker(ticker)
        history = stock.history(start=buy_date, end=today)
        historical_data[ticker] = history

        try:
            current_prices.append(history["Close"].iloc[-1])
        except:
            current_prices.append(None)

    # Berechnungen
    df["Current Price"] = current_prices
    df["Value"] = df["Current Price"] * df["Shares"]
    df["Profit/Loss"] = (df["Current Price"] - df["Buy Price"]) * df["Shares"]

    total_value = df["Value"].sum()
    total_gain = df["Profit/Loss"].sum()

    # ============ 📈 Portfolio Overview ============
    if menu == "📈 Portfolio Overview":
        st.title("📈 Portfolio Overview")
        st.dataframe(df)

        st.subheader("💰 Totals")
        st.write(f"**Total Portfolio Value:** €{round(total_value, 2)}")
        st.write(f"**Total Profit/Loss:** €{round(total_gain, 2)}")

        st.subheader("📊 Allocation by Value")
        fig, ax = plt.subplots()
        ax.pie(df["Value"], labels=df["Ticker"], autopct='%1.1f%%', startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

    # ============ 📉 Performance & Risk Analytics ============
    elif menu == "📉 Performance & Risk Analytics":
        st.title("📉 Performance and Risk Analytics")

        st.markdown("""
        This section shows:
        - Per Asset: Volatility, Max Drawdown, Beta vs S&P500
        - Whole Portfolio: Sharpe Ratio, Sortino Ratio, Max Drawdown, CAGR
        """)

        returns = []
        start_date = df["Buy Date"].min().date()
        benchmark = yf.Ticker("^GSPC").history(start=start_date, end=datetime.today().date())["Close"].pct_change()

        for ticker, history in historical_data.items():
            history["Return"] = history["Close"].pct_change()
            returns.append(history["Return"])

            volatility = history["Return"].std() * np.sqrt(252)
            max_dd = calculate_max_drawdown(history["Return"])

            aligned = pd.concat([history["Return"], benchmark], axis=1).dropna()
            if not aligned.empty:
                slope, *_ = linregress(aligned.iloc[:, 1], aligned.iloc[:, 0])
            else:
                slope = np.nan

            with st.expander(f"📌 {ticker} Metrics"):
                st.write("**Volatility (Annualized):**", round(volatility, 4))
                st.write("**Max Drawdown:**", round(max_dd, 4))
                st.write("**Beta vs S&P500:**", round(slope, 4))

        if returns:
            combined_returns = pd.concat(returns, axis=1).mean(axis=1)

            sharpe = combined_returns.mean() / combined_returns.std() * np.sqrt(252)
            downside = combined_returns[combined_returns < 0].std() * np.sqrt(252)
            sortino = combined_returns.mean() / downside if downside else np.nan
            max_dd = calculate_max_drawdown(combined_returns)

            days = (combined_returns.index[-1] - combined_returns.index[0]).days
            years = days / 365.25
            cumulative_return = (1 + combined_returns).prod()
            cagr = calculate_cagr(1, cumulative_return, years)

            st.subheader("📦 Portfolio Summary")
            st.metric("Sharpe Ratio", round(sharpe, 3))
            st.metric("Sortino Ratio", round(sortino, 3))
            st.metric("Max Drawdown", round(max_dd, 3))
            st.metric("CAGR", f"{round(cagr * 100, 2)}%")

            st.subheader("📈 Cumulative Return")
            cumulative = (1 + combined_returns).cumprod()
            fig2, ax2 = plt.subplots()
            ax2.plot(cumulative.index, cumulative.values)
            ax2.set_title("Cumulative Portfolio Return")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Cumulative Return")
            st.pyplot(fig2)

# ------------------- Helper Functions -------------------
def calculate_cagr(start_value, end_value, periods):
    return (end_value / start_value) ** (1 / periods) - 1

def calculate_max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
