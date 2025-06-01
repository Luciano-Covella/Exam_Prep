import pandas as pd
import streamlit as st
import yfinance as yf
from io import StringIO
from datetime import datetime

def load_portfolio_data(file):
    if not file:
        st.info("Please upload a CSV first.")
        return None

    try:
        decoded = StringIO(file.decode("utf-8"))
        df = pd.read_csv(decoded)
        df["Buy Date"] = pd.to_datetime(df["Buy Date"])
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

    current_prices = []
    dividends_received = []
    today = datetime.today()

    for _, row in df.iterrows():
        stock = yf.Ticker(row["Ticker"])

        # Load current price
        try:
            history = stock.history(start=row["Buy Date"], end=today)
            history.index = history.index.tz_localize(None)
            price = history["Close"].iloc[-1]
        except:
            price = None
        current_prices.append(price)

        # Calculating dividends
        divs = stock.dividends
        if not divs.empty:
            divs.index = divs.index.tz_localize(None)
            filtered = divs[(divs.index >= row["Buy Date"]) & (divs.index <= today)]
            total_divs = filtered.sum() * row["Shares"]
        else:
            total_divs = 0.0
        dividends_received.append(total_divs)

    # Compute columns in the DataFrame
    df["Current Price"] = current_prices
    df["Current"] = df["Current Price"]  
    df["Value"] = df["Current"] * df["Shares"]
    df["Profit/Loss"] = (df["Current"] - df["Buy Price"]) * df["Shares"]
    df["Dividends"] = dividends_received
    df["Total Return"] = df["Profit/Loss"] + df["Dividends"]

    return df


def get_historical_data(df):
    historical_data = {}
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        stock = yf.Ticker(ticker)
        history = stock.history(start=row["Buy Date"], end=datetime.today())
        history.index = history.index.tz_localize(None)
        historical_data[ticker] = history
    return historical_data
