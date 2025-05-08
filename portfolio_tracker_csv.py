# Import required libraries
import streamlit as st                    # Web interface library
import pandas as pd                       # Data handling and CSV reading
import yfinance as yf                     # Getting financial data (stock prices)
import matplotlib.pyplot as plt           # Creating plots and charts
import numpy as np                        # Mathematical calculations (returns, ratios)
from datetime import datetime             # Handling dates
from scipy.stats import linregress        # For calculating Beta (regression)

# Configure Streamlit page
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("📊 Portfolio Analyzer (Snapshot + Analytics + Buy Date + Advanced Metrics)")

# User file upload
st.write("Upload your Portfolio CSV file (Ticker, Shares, Buy Price, Buy Date):")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Helper function to calculate CAGR (Compound Annual Growth Rate)
def calculate_cagr(start_value, end_value, periods):
    return (end_value / start_value) ** (1 / periods) - 1

# Helper function to calculate Maximum Drawdown
def calculate_max_drawdown(series):
    cumulative = (1 + series).cumprod()       # Calculate cumulative returns
    peak = cumulative.cummax()                # Find the historical peaks
    drawdown = (cumulative - peak) / peak     # Calculate drawdowns from peaks
    return drawdown.min()                     # Return the largest drawdown (most negative)

# If the user uploaded a CSV file, proceed
if uploaded_file is not None:
    # Read the CSV into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Validate if required columns exist
    if not all(col in df.columns for col in ["Ticker", "Shares", "Buy Price", "Buy Date"]):
        st.error("ERROR: The CSV must contain: Ticker, Shares, Buy Price, Buy Date.")
    else:
        # Prepare variables for storing results
        current_prices = []
        historical_data = {}

        st.write("📥 Fetching live prices and historical data...")

        # Iterate through each asset in the portfolio
        for index, row in df.iterrows():
            ticker = row["Ticker"]
            buy_date = pd.to_datetime(row["Buy Date"])
            today = datetime.today()

            # Get stock data from Yahoo Finance starting from Buy Date
            stock = yf.Ticker(ticker)
            history = stock.history(start=buy_date, end=today)

            # Store the historical data for later analysis
            historical_data[ticker] = history

            # Get the latest closing price (most recent price)
            try:
                price = history["Close"].iloc[-1]
                current_prices.append(price)
            except:
                current_prices.append(None)

        # Add current prices to DataFrame
        df["Current Price"] = current_prices
        df["Value"] = df["Current Price"] * df["Shares"]
        df["Profit/Loss"] = (df["Current Price"] - df["Buy Price"]) * df["Shares"]

        total_value = df["Value"].sum()
        total_gain = df["Profit/Loss"].sum()

        # ----------------------
        # Portfolio Overview
        # ----------------------
        st.header("📌 Portfolio Overview")
        st.dataframe(df)

        st.write(f"**Total Portfolio Value:** € {round(total_value, 2)}")
        st.write(f"**Total Profit/Loss:** € {round(total_gain, 2)}")

        # Pie Chart for Allocation
        st.write("### 📊 Portfolio Allocation")
        fig, ax = plt.subplots()
        ax.pie(df["Value"], labels=df["Ticker"], autopct='%1.1f%%', startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

        # ----------------------
        # Performance & Risk Analytics
        # ----------------------
        st.header("📈 Performance & Risk Analytics (since Buy Date)")

        returns = []
       # Calculate Benchmark returns (SP500, "^GSPC")
        start_date = pd.to_datetime(df["Buy Date"]).min().date()
        benchmark = yf.Ticker("^GSPC").history(start=start_date, end=datetime.today().date())["Close"].pct_change()


        # For each stock, calculate returns and metrics
        for ticker, history in historical_data.items():
            history["Return"] = history["Close"].pct_change()
            returns.append(history["Return"])

            # Volatility (Standard Deviation of returns, annualized)
            volatility = history["Return"].std() * np.sqrt(252)

            st.write(f"**{ticker} Volatility (annualized):** {round(volatility, 4)}")

            # Max Drawdown
            max_dd = calculate_max_drawdown(history["Return"])
            st.write(f"**{ticker} Max Drawdown:** {round(max_dd, 4)}")

            # Beta Calculation (vs. SP500 benchmark)
            aligned = pd.concat([history["Return"], benchmark], axis=1).dropna()
            if not aligned.empty:
                slope, intercept, r_value, p_value, std_err = linregress(aligned.iloc[:, 1], aligned.iloc[:, 0])
                st.write(f"**{ticker} Beta vs SP500:** {round(slope, 4)}")

        # Combine returns of all assets (equal weighted)
        if returns:
            combined_returns = pd.concat(returns, axis=1)
            combined_returns.columns = historical_data.keys()
            portfolio_return = combined_returns.mean(axis=1)

            # Sharpe Ratio (Risk-adjusted return)
            sharpe_ratio = portfolio_return.mean() / portfolio_return.std() * np.sqrt(252)
            st.write(f"**Portfolio Sharpe Ratio:** {round(sharpe_ratio, 3)}")

            # Sortino Ratio (penalizing only negative returns)
            downside = portfolio_return[portfolio_return < 0].std() * np.sqrt(252)
            sortino_ratio = portfolio_return.mean() / downside if downside != 0 else np.nan
            st.write(f"**Portfolio Sortino Ratio:** {round(sortino_ratio, 3)}")

            # Portfolio Max Drawdown
            portfolio_max_dd = calculate_max_drawdown(portfolio_return)
            st.write(f"**Portfolio Max Drawdown:** {round(portfolio_max_dd, 4)}")

            # CAGR calculation
            days = (portfolio_return.index[-1] - portfolio_return.index[0]).days
            years = days / 365.25
            cumulative_return = (1 + portfolio_return).prod()
            cagr = calculate_cagr(1, cumulative_return, years)
            st.write(f"**Portfolio CAGR:** {round(cagr, 4)}")

            # Portfolio Performance Chart
            st.write("### 📅 Portfolio Performance (Cumulative)")
            cumulative = (1 + portfolio_return).cumprod()
            fig2, ax2 = plt.subplots()
            ax2.plot(cumulative.index, cumulative.values)
            ax2.set_title("Cumulative Portfolio Return")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Cumulative Return")
            st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to start.")
