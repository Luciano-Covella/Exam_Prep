import streamlit as st
import streamlit as st
import matplotlib.pyplot as plt

theme = st.session_state.get("theme", "Light")
plt.style.use("dark_background" if theme == "Dark" else "default")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from utils.calculations import calculate_cagr, calculate_max_drawdown
import yfinance as yf
import seaborn as sns

def get_price_history(ticker, start, end):
    try:
        data = yf.Ticker(ticker).history(start=start, end=end)
        return data["Close"].pct_change().dropna()
    except:
        return pd.Series(dtype=float)

def show_performance():
    st.title("📉 Performance & Risk Analytics")

    df = st.session_state.get("portfolio_df", None)
    if df is None:
        st.warning("Please upload a portfolio CSV first.")
        return

    start_date = df["Buy Date"].min().date()
    end_date = datetime.today().date()

    tickers = df["Ticker"].unique().tolist()
    returns_data = {}
    dividends_data = {}

    benchmark = get_price_history("^GSPC", start_date, end_date)

    for ticker in tickers:
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
        hist.index = hist.index.tz_localize(None)
        returns = hist["Close"].pct_change().dropna()
        returns_data[ticker] = returns

        dividends = yf.Ticker(ticker).dividends
        dividends.index = dividends.index.tz_localize(None)
        mask = (dividends.index >= pd.to_datetime(start_date)) & (dividends.index <= pd.to_datetime(end_date))
        filtered = dividends.loc[mask]
        dividends_data[ticker] = filtered.groupby(filtered.index.year).sum() if not filtered.empty else pd.Series(dtype=float)

    portfolio_returns = pd.concat(returns_data.values(), axis=1).mean(axis=1).dropna()

    # --- Computing metrics ---
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    downside_std = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
    sortino = portfolio_returns.mean() / downside_std if downside_std > 0 else 0
    max_dd = calculate_max_drawdown(portfolio_returns)
    days = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days
    cagr = calculate_cagr(1, (1 + portfolio_returns).prod(), days / 365.25)

    total_value = df["Shares"] * df["Current"]
    income_total = 0.0
    for ticker in tickers:
        divs = dividends_data.get(ticker, pd.Series(dtype=float))
        if not divs.empty:
            shares = df.loc[df["Ticker"] == ticker, "Shares"].iloc[0]
            income_total += divs.sum() * shares
    income_yield = income_total / total_value.sum() if total_value.sum() > 0 else 0

    # --- Visualization ---
    st.subheader("Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return (%)", f"{(1 + portfolio_returns).prod() - 1:.2%}")
    col2.metric("CAGR (%)", f"{cagr * 100:.2f}%")
    col3.metric("Volatility", f"{portfolio_returns.std() * np.sqrt(252):.2f}")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Sortino Ratio", f"{sortino:.2f}")
    col6.metric("Max Drawdown", f"{max_dd:.2f}")
    col7.metric("Income Yield (%)", f"{income_yield * 100:.2f}%")

    # ✅ Benchmark-Beta
    benchmark.index = benchmark.index.tz_localize(None)
    portfolio_returns.index = portfolio_returns.index.tz_localize(None)
    aligned = pd.concat([benchmark, portfolio_returns], axis=1).dropna()
    beta = linregress(aligned.iloc[:, 0], aligned.iloc[:, 1]).slope
    col8.metric("Benchmark Beta", f"{beta:.2f}")

    # --- Cumulative Return ---
    st.subheader("Cumulative Return")
    fig, ax = plt.subplots()
    ax.plot((1 + portfolio_returns).cumprod(), label="Portfolio", linewidth=2)
    ax.plot((1 + benchmark).cumprod(), label="S&P 500", linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # --- Dividends Chart ---
    st.subheader("Received Dividends")
    dividend_df = pd.DataFrame({
        ticker: divs * df.loc[df["Ticker"] == ticker, "Shares"].iloc[0]
        for ticker, divs in dividends_data.items()
    }).fillna(0)

    if not dividend_df.empty:
        fig2, ax2 = plt.subplots()
        dividend_df.plot(kind="bar", stacked=True, ax=ax2)
        ax2.set_ylabel("Dividends (€)")
        st.pyplot(fig2)
    else:
        st.info("No dividend data available.")

    # --- Correlation ---
    st.subheader("Return Correlation Matrix")
    returns_matrix = pd.DataFrame(returns_data).fillna(0)
    corr = returns_matrix.corr()
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, cmap="viridis", annot=True, fmt=".2f", ax=ax3)
    st.pyplot(fig3)
