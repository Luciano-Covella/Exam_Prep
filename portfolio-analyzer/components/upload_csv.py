import streamlit as st
import pandas as pd
from utils.data_loader import load_portfolio_data
from io import StringIO

REQUIRED_COLUMNS = ["Ticker", "Shares", "Buy Price", "Buy Date"]

def show_upload():
    st.title("📂 Upload CSV")
    st.info("Please upload a CSV file with the following columns:\n\n"
            "`Ticker`, `Shares`, `Buy Price`, `Buy Date`")

    uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            content = uploaded_file.read()
            decoded = StringIO(content.decode("utf-8"))
            preview_df = pd.read_csv(decoded)

            # Validierung der Spalten
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in preview_df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return

            # ✅ Portfolio-Daten mit Preisen & Dividenden laden
            df = load_portfolio_data(content)
            if df is None:
                st.error("❌ Failed to process file.")
                return

            # Speichern im Session State
            st.session_state["portfolio_file"] = content
            st.session_state["portfolio_filename"] = uploaded_file.name
            st.session_state["portfolio_df"] = df

            st.success("✅ File uploaded and processed successfully!")
            st.write("Preview:", df.head())

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
