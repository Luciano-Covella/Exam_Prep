import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Tracker", layout="centered")
st.title("📊 Portfolio Tracker mit CSV Upload")

st.write("Bitte lade deine Portfolio CSV hoch (Ticker, Menge, Kaufpreis):")

uploaded_file = st.file_uploader("CSV Datei hochladen", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Datei erfolgreich geladen!")

        # Prüfen ob notwendige Spalten vorhanden sind
        if not all(col in df.columns for col in ["Ticker", "Menge", "Kaufpreis"]):
            st.error("Fehler: Die CSV-Datei muss die Spalten 'Ticker', 'Menge' und 'Kaufpreis' enthalten.")
        else:
            # Live Preise abrufen
            prices = []
            for ticker in df["Ticker"]:
                try:
                    stock = yf.Ticker(ticker)
                    price = stock.history(period="1d")["Close"].iloc[-1]
                    prices.append(price)
                except:
                    prices.append(None)

            df["Aktueller Preis"] = prices
            df["Wert"] = df["Aktueller Preis"] * df["Menge"]
            df["Gewinn/Verlust"] = (df["Aktueller Preis"] - df["Kaufpreis"]) * df["Menge"]

            st.write("### 📈 Portfolio Übersicht")
            st.dataframe(df)

            total_value = df["Wert"].sum()
            total_gain = df["Gewinn/Verlust"].sum()

            st.write(f"**Gesamtwert des Portfolios:** € {round(total_value, 2)}")
            st.write(f"**Gesamter Gewinn/Verlust:** € {round(total_gain, 2)}")

            # Pie Chart
            st.write("### 📊 Portfolio Verteilung")
            fig, ax = plt.subplots()
            ax.pie(df["Wert"], labels=df["Ticker"], autopct='%1.1f%%', startangle=140)
            ax.axis("equal")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Fehler beim Verarbeiten der Datei: {e}")

else:
    st.info("Bitte lade eine CSV-Datei hoch, um zu starten.") 
