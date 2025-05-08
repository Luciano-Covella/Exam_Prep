import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("📊 Portfolio Analyzer (Snapshot + Analytics + Buy Date)")

st.write("Lade dein Portfolio CSV hoch (Ticker, Menge, Kaufpreis, Kaufdatum):")

uploaded_file = st.file_uploader("CSV Datei hochladen", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in ["Ticker", "Menge", "Kaufpreis", "Kaufdatum"]):
        st.error("Fehler: Die CSV-Datei muss die Spalten 'Ticker', 'Menge', 'Kaufpreis' und 'Kaufdatum' enthalten.")
    else:
        prices = []
        histories = {}

        st.write("📥 Live Preise werden geladen...")

        for index, row in df.iterrows():
            ticker = row["Ticker"]
            buy_date = pd.to_datetime(row["Kaufdatum"])
            today = datetime.today()

            stock = yf.Ticker(ticker)

            hist = stock.history(start=buy_date, end=today)
            histories[ticker] = hist

            try:
                price = hist["Close"].iloc[-1]
                prices.append(price)
            except:
                prices.append(None)

        df["Aktueller Preis"] = prices
        df["Wert"] = df["Aktueller Preis"] * df["Menge"]
        df["Gewinn/Verlust"] = (df["Aktueller Preis"] - df["Kaufpreis"]) * df["Menge"]

        total_value = df["Wert"].sum()
        total_gain = df["Gewinn/Verlust"].sum()

        st.header("📌 Portfolio Übersicht")
        st.dataframe(df)

        st.write(f"**Gesamtwert:** € {round(total_value, 2)}")
        st.write(f"**Gesamter Gewinn/Verlust:** € {round(total_gain, 2)}")

        # Pie Chart
        st.write("### 📊 Portfolio Verteilung")
        fig, ax = plt.subplots()
        ax.pie(df["Wert"], labels=df["Ticker"], autopct='%1.1f%%', startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

        # === Erweiterung: Analytics ===
        st.header("📈 Performance & Risiko Analyse (ab Kaufdatum)")

        returns = []

        for ticker, hist in histories.items():
            hist["Return"] = hist["Close"].pct_change()
            returns.append(hist["Return"])

            st.write(f"**{ticker} Volatilität (seit Kaufdatum):** {round(hist['Return'].std() * np.sqrt(252), 4)}")

        if returns:
            combined_returns = pd.concat(returns, axis=1)
            combined_returns.columns = histories.keys()
            portfolio_return = combined_returns.mean(axis=1)

            sharpe_ratio = portfolio_return.mean() / portfolio_return.std() * np.sqrt(252)

            st.write(f"**Portfolio Sharpe Ratio (seit Kaufdatum):** {round(sharpe_ratio, 3)}")

            st.write("### 📅 Portfolio Performance Verlauf (seit Kaufdatum)")
            portfolio_cum_return = (1 + portfolio_return).cumprod()

            fig2, ax2 = plt.subplots()
            ax2.plot(portfolio_cum_return.index, portfolio_cum_return.values)
            ax2.set_title("Portfolio kumulierte Performance")
            ax2.set_xlabel("Datum")
            ax2.set_ylabel("Kumulierte Rendite")
            st.pyplot(fig2)

else:
    st.info("Bitte lade eine CSV-Datei hoch.")
