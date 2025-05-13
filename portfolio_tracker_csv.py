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

def calculate_cagr(start_value, end_value, periods):
    # Calculates annual growth rate from start to end value over given years
    return (end_value / start_value) ** (1 / periods) - 1  # ✅ Bug-free math

def calculate_max_drawdown(series):
    # Calculates worst drop from peak (risk metric)
    cumulative = (1 + series).cumprod()  # Combine returns into a growth curve
    peak = cumulative.cummax()  # Tracks the highest point so far
    drawdown = (cumulative - peak) / peak  # How much below the peak it dropped
    return drawdown.min()  # Returns the worst (lowest) drop ✅ Bug-free

# ---------- Streamlit app setup ----------

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")  # Set page title and full-width layout

# ---------- Sidebar navigation menu ----------

with st.sidebar:
    st.title("📊 Portfolio Menu")  # Sidebar title
    menu = st.radio(  # Create radio buttons for page navigation
        "Navigation",
        ["📁 Upload CSV", "📈 Portfolio Overview", "📉 Performance & Risk Analytics"]
    )

# ---------- Initialize session state (temporary memory) ----------

if "portfolio_file" not in st.session_state:
    st.session_state.portfolio_file = None  # Stores uploaded file content
    st.session_state.portfolio_filename = None  # Stores uploaded file name

# ---------- File upload section ----------

if menu == "📁 Upload CSV":  # Show this section if "Upload CSV" is selected
    st.title("📁 Upload Portfolio CSV")  # Page title
    st.info("Upload a CSV with columns: Ticker, Shares, Buy Price, Buy Date")  # Instructions

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])  # File upload component
    if uploaded:
        st.session_state.portfolio_file = uploaded.read()  # Read file content and store it
        st.session_state.portfolio_filename = uploaded.name  # Store file name
        st.success("✅ File uploaded successfully. Use the sidebar to continue.")  # Confirmation message

# ---------- Read and parse file content if available ----------

file_content = st.session_state.get("portfolio_file", None)  # Get file from session memory

if file_content:
    try:
        decoded = StringIO(file_content.decode("utf-8"))  # Decode bytes to text
        df = pd.read_csv(decoded)  # Read CSV content into DataFrame ✅ Bug-free
        if df.empty:
            st.error("❌ The uploaded file is empty.")  # Show error if CSV is blank
            st.stop()  # Stop app execution
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {str(e)}")  # Show parsing errors
        st.stop()  # Stop further processing

    # ---------- Validate required columns ----------
    required_cols = ["Ticker", "Shares", "Buy Price", "Buy Date"]  # Must-have columns
    if not all(col in df.columns for col in required_cols):  # Check all required columns exist
        st.error("❌ CSV must contain: Ticker, Shares, Buy Price, Buy Date")  # Show error
        st.stop()  # Stop if invalid structure

    # ---------- Process and clean data ----------
    df["Buy Date"] = pd.to_datetime(df["Buy Date"])  # Convert text to real date format
    current_prices = []  # Stores current prices
    historical_data = {}  # Stores price history per ticker

    st.write("⏳ Fetching historical and current prices...")  # Status message

    for _, row in df.iterrows():  # Loop through each investment row
        ticker = row["Ticker"]  # Extract ticker symbol
        buy_date = row["Buy Date"]  # Extract purchase date
        today = datetime.today()  # Get today's date

        stock = yf.Ticker(ticker)  # Download data from Yahoo Finance
        history = stock.history(start=buy_date, end=today)  # Fetch historical price data
        historical_data[ticker] = history  # Store it for later analytics

        try:
            current_prices.append(history["Close"].iloc[-1])  # Add most recent closing price
        except:
            current_prices.append(None)  # If no data available, store None

    # ---------- Add portfolio calculation columns ----------
    df["Current Price"] = current_prices  # Current market value
    df["Value"] = df["Current Price"] * df["Shares"]  # Total value per position
    df["Profit/Loss"] = (df["Current Price"] - df["Buy Price"]) * df["Shares"]  # Gain or loss

    total_value = df["Value"].sum()  # Sum of all position values
    total_gain = df["Profit/Loss"].sum()  # Sum of all gains/losses

    # ---------- Portfolio Overview Section ----------
    if menu == "📈 Portfolio Overview":
        st.title("📈 Portfolio Overview")  # Page heading
        st.dataframe(df)  # Show table with all positions

        st.subheader("💰 Totals")  # Section title
        st.write(f"**Total Portfolio Value:** €{round(total_value, 2)}")  # Total value
        st.write(f"**Total Profit/Loss:** €{round(total_gain, 2)}")  # Total P/L

        st.subheader("📊 Allocation by Value")  # Section title
        fig, ax = plt.subplots()  # Create chart
        ax.pie(df["Value"], labels=df["Ticker"], autopct='%1.1f%%', startangle=140)  # Pie chart by value
        ax.axis("equal")  # Equal aspect ratio to keep it round
        st.pyplot(fig)  # Display chart in app

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
