# Import required libraries
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import linregress

# Configure Streamlit page
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# Sidebar menu
menu = st.sidebar.radio(
    "Navigation",
    ["📄 Upload CSV", "📊 Portfolio Overview", "📈 Performance & Risk Analytics"]
)

# Helper function: Calculate CAGR (Compound Annual Growth Rate)
def calculate_cagr(start_value, end_value, periods):
    return (end_value / start_value) ** (1 / periods) - 1

# Helper function: Calculate maximum drawdown from a return series
def calculate_max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# File uploader section
uploaded_file = None
if menu == "📄 Upload CSV":
    st.title("📄 Upload Portfolio CSV")
    st.write("Upload your Portfolio CSV file (columns: Ticker, Shares, Buy Price, Buy Date)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# If a file is uploaded and valid
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    required_cols = ["Ticker", "Shares", "Buy Price", "Buy Date"]
    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain: Ticker, Shares, Buy Price, Buy Date")
    else:
        df["Buy Date"] = pd.to_datetime(df["Buy Date"])

        current_prices = []
        historical_data = {}

        st.write("📅 Fetching historical and current prices...")

        # Fetch prices for each asset
        for _, row in df.iterrows():
            ticker = row["Ticker"]
            buy_date = row["Buy Date"]
            today = datetime.today()

            stock = yf.Ticker(ticker)
            history = stock.history(start=buy_date, end=today)
            historical_data[ticker] = history

            try:
                price = history["Close"].iloc[-1]
                current_prices.append(price)
            except:
                current_prices.append(None)

        # Calculate performance
        df["Current Price"] = current_prices
        df["Value"] = df["Current Price"] * df["Shares"]
        df["Profit/Loss"] = (df["Current Price"] - df["Buy Price"]) * df["Shares"]

        total_value = df["Value"].sum()
        total_gain = df["Profit/Loss"].sum()

        # ========== Portfolio Overview ==========
        if menu == "📊 Portfolio Overview":
            st.title("📊 Portfolio Overview")
            st.dataframe(df)

            st.write(f"**Total Portfolio Value:** €{round(total_value, 2)}")
            st.write(f"**Total Profit/Loss:** €{round(total_gain, 2)}")

            st.write("### 📈 Allocation by Value")
            fig, ax = plt.subplots()
            ax.pie(df["Value"], labels=df["Ticker"], autopct='%1.1f%%', startangle=140)
            ax.axis("equal")
            st.pyplot(fig)

        # ========== Performance and Risk Analytics ==========
        elif menu == "📈 Performance & Risk Analytics":
            st.title("📈 Performance and Risk Analytics Overview")
            st.write("""
            This section shows performance indicators and risk metrics:
            - Per asset: Volatility, Max Drawdown, Beta to SP500
            - Entire portfolio: Sharpe, Sortino, Max Drawdown, CAGR
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

                with st.expander(f"📉 {ticker} - Risk Metrics"):
                    st.write("**Volatility (Annualized):**", round(volatility, 4))
                    st.write("**Max Drawdown:**", round(max_dd, 4))
                    st.write("**Beta vs SP500:**", round(slope, 4))

            if returns:
                combined_returns = pd.concat(returns, axis=1).mean(axis=1)

                sharpe_ratio = combined_returns.mean() / combined_returns.std() * np.sqrt(252)
                downside = combined_returns[combined_returns < 0].std() * np.sqrt(252)
                sortino_ratio = combined_returns.mean() / downside if downside else np.nan
                portfolio_max_dd = calculate_max_drawdown(combined_returns)

                days = (combined_returns.index[-1] - combined_returns.index[0]).days
                years = days / 365.25
                cumulative_return = (1 + combined_returns).prod()
                cagr = calculate_cagr(1, cumulative_return, years)

                st.write("### 📦 Portfolio Performance Summary")
                st.metric("Sharpe Ratio", round(sharpe_ratio, 3))
                st.metric("Sortino Ratio", round(sortino_ratio, 3))
                st.metric("Max Drawdown", round(portfolio_max_dd, 3))
                st.metric("CAGR", f"{round(cagr * 100, 2)}%")

                st.write("### 📆 Portfolio Cumulative Performance")
                cumulative = (1 + combined_returns).cumprod()
                fig2, ax2 = plt.subplots()
                ax2.plot(cumulative.index, cumulative.values)
                ax2.set_title("Cumulative Portfolio Return")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Cumulative Return")
                st.pyplot(fig2)
