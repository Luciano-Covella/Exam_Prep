import streamlit as st
import matplotlib.pyplot as plt

theme = st.session_state.get("theme", "Light")
plt.style.use("dark_background" if theme == "Dark" else "default")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_portfolio_data
from utils.calculations import calculate_cagr, calculate_max_drawdown


def show_overview():
    st.title("📈 Portfolio Overview")

    file_content = st.session_state.get("portfolio_file", None)
    if not file_content:
        st.warning("Please upload a portfolio CSV file first.")
        return

    df = load_portfolio_data(file_content)
    if df is None:
        st.error("Failed to load portfolio data.")
        return

    total_value = df["Value"].sum()
    total_gain = df["Profit/Loss"].sum()

    col1, col2 = st.columns(2)
    col1.metric("Total Value", f"€{total_value:,.2f}")
    col2.metric("Total P/L", f"€{total_gain:,.2f}")

    # --- Allocation by Value ---
    st.subheader("📊 Allocation by Value")
    fig1, ax1 = plt.subplots()
    ax1.pie(df["Value"], labels=df["Ticker"], autopct='%1.1f%%', startangle=140)
    ax1.axis("equal")
    st.pyplot(fig1)

    # --- Allocation by Sector (dummy sectors if not present) ---
    st.subheader("📊 Allocation by Sector")
    dummy_sectors = {
        "AAPL": "Technology", "TSLA": "Consumer Cyclical", "BTC-USD": "N/A",
        "ETH-USD": "N/A", "GOOGL": "Communication Services", "NVDA": "Technology"
    }
    df["Sector"] = df["Ticker"].map(dummy_sectors).fillna("N/A")
    sector_summary = df.groupby("Sector")["Value"].sum()
    fig2, ax2 = plt.subplots()
    ax2.pie(sector_summary, labels=sector_summary.index, autopct='%1.1f%%')
    ax2.axis("equal")
    st.pyplot(fig2)

    # --- Footer Info ---
    st.sidebar.write("Last updated:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.sidebar.write("File:", st.session_state.get("portfolio_filename", "-"))
