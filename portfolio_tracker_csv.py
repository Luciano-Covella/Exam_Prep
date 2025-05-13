# Import required libraries
import streamlit as st                    # Web interface library
import pandas as pd                       # Data handling and CSV reading
import yfinance as yf                     # Fetching financial and market data
import matplotlib.pyplot as plt           # Plotting and visualization
import numpy as np                        # Numerical computations (returns, ratios)
from datetime import datetime             # Working with date and time
from scipy.stats import linregress        # Statistical regression (used to calculate Beta)

# Configure Streamlit page
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# Title of the web app
st.title("\U0001F4CA Portfolio Analyzer (Snapshot + Analytics + Metrics)")

# File uploader prompt
st.write("Upload your Portfolio CSV file (columns: Ticker, Shares, Buy Price, Buy Date)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Helper function: Calculate CAGR (Compound Annual Growth Rate)
def calculate_cagr(start_value, end_value, periods):
    return (end_value / start_value) ** (1 / periods) - 1

# Helper function: Calculate maximum drawdown from a return series
def calculate_max_drawdown(series):
    cumulative = (1 + series).cumprod()           # Calculate cumulative return curve
    peak = cumulative.cummax()                    # Find rolling maximum
    drawdown = (cumulative - peak) / peak         # Calculate drawdown from peak
    return drawdown.min()                         # Return the worst (lowest) drawdown

# If file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Load CSV data into DataFrame

    # Check if required columns exist
    if not all(col in df.columns for col in ["Ticker", "Shares", "Buy Price", "Buy Date"]):
        st.error("CSV must contain: Ticker, Shares, Buy Price, Buy Date")
    else:
        # Convert Buy Date column to datetime
        df["Buy Date"] = pd.to_datetime(df["Buy Date"])

        # Prepare containers
        current_prices = []             # Stores latest price per asset
        historical_data = {}            # Stores historical prices for each asset

        st.write("\U0001F4C5 Fetching historical and current prices...")

        # Loop through portfolio positions
        for index, row in df.iterrows():
            ticker = row["Ticker"]
            buy_date = row["Buy Date"]
            today = datetime.today()

            stock = yf.Ticker(ticker)                       # Load ticker
            history = stock.history(start=buy_date, end=today)  # Get data from buy date to today
            historical_data[ticker] = history               # Save full history

            # Try to get the latest price
            try:
                price = history["Close"].iloc[-1]           # Last closing price
                current_prices.append(price)
            except:
                current_prices.append(None)                 # In case of data error

        # Calculate live stats
        df["Current Price"] = current_prices
        df["Value"] = df["Current Price"] * df["Shares"]
        df["Profit/Loss"] = (df["Current Price"] - df["Buy Price"]) * df["Shares"]

        total_value = df["Value"].sum()
        total_gain = df["Profit/Loss"].sum()

        # ----------------------
        # Portfolio Overview
        # ----------------------
        st.header("\U0001F4CB Portfolio Overview")
        st.dataframe(df)

        # Display totals
        st.write(f"**Total Portfolio Value:** €{round(total_value, 2)}")
        st.write(f"**Total Profit/Loss:** €{round(total_gain, 2)}")

        # Pie chart: allocation by value
        st.write("### \U0001F4C8 Allocation by Value")
        fig, ax = plt.subplots()
        ax.pie(df["Value"], labels=df["Ticker"], autopct='%1.1f%%', startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

        # ----------------------
        # Performance & Risk Analytics
        # ----------------------
        st.write("## \U0001F4CA Performance and Risk Analytics Overview")
        st.write("""
        This section shows performance indicators and risk metrics:
        - Per asset: Volatility, Max Drawdown, Beta to SP500
        - Entire portfolio: Sharpe, Sortino, Max Drawdown, CAGR
        """)

        returns = []

        # Benchmark: SP500 (^GSPC)
        start_date = df["Buy Date"].min().date()    # earliest buy date in portfolio
        benchmark = yf.Ticker("^GSPC").history(start=start_date, end=datetime.today().date())["Close"].pct_change()

        # Per-asset analytics using expanders
        for ticker, history in historical_data.items():
            history["Return"] = history["Close"].pct_change()
            returns.append(history["Return"])

            # Volatility (annualized)
            volatility = history["Return"].std() * np.sqrt(252)

            # Max Drawdown
            max_dd = calculate_max_drawdown(history["Return"])

            # Beta vs SP500 
            aligned = pd.concat([history["Return"], benchmark], axis=1).dropna()
            if not aligned.empty:
                slope, *_ = linregress(aligned.iloc[:, 1], aligned.iloc[:, 0])
            else:
                slope = np.nan

            # Display inside expander
            with st.expander(f"\U0001F4C9 {ticker} - Risk Metrics"):
                st.write("**Volatility (Annualized):**", round(volatility, 4))
                st.write("**Max Drawdown:**", round(max_dd, 4))
                st.write("**Beta vs SP500:**", round(slope, 4))

        # ----------------------
        # Portfolio-wide analytics
        # ----------------------
        if returns:
            combined_returns = pd.concat(returns, axis=1).mean(axis=1)

            # Sharpe Ratio
            sharpe_ratio = combined_returns.mean() / combined_returns.std() * np.sqrt(252)

            # Sortino Ratio (downside deviation)
            downside = combined_returns[combined_returns < 0].std() * np.sqrt(252)
            sortino_ratio = combined_returns.mean() / downside if downside else np.nan

            # Max Drawdown
            portfolio_max_dd = calculate_max_drawdown(combined_returns)

            # CAGR
            days = (combined_returns.index[-1] - combined_returns.index[0]).days
            years = days / 365.25
            cumulative_return = (1 + combined_returns).prod()
            cagr = calculate_cagr(1, cumulative_return, years)

            # Display metrics
            st.write("### \U0001F4E6 Portfolio Performance Summary")
            st.metric("Sharpe Ratio", round(sharpe_ratio, 3))
            st.metric("Sortino Ratio", round(sortino_ratio, 3))
            st.metric("Max Drawdown", round(portfolio_max_dd, 3))
            st.metric("CAGR", f"{round(cagr * 100, 2)}%")

            # Cumulative performance chart
            st.write("### \U0001F4C5 Portfolio Cumulative Performance")
            cumulative = (1 + combined_returns).cumprod()
            fig2, ax2 = plt.subplots()
            ax2.plot(cumulative.index, cumulative.values)
            ax2.set_title("Cumulative Portfolio Return")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Cumulative Return")
            st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to continue.")