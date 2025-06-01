# app.py (Launcher for modular Streamlit app)
import streamlit as st
from components.upload_csv import show_upload
from components.overview import show_overview
from components.performance import show_performance

# === Global Session State Defaults ===
if "portfolio_file" not in st.session_state:
    st.session_state["portfolio_file"] = None

if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = None

if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

# === Page Settings ===
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# === Sidebar Navigation ===
from components.sidebar import show_sidebar
menu_option = show_sidebar()

# === Routing ===
if menu_option == "📁 Upload CSV":
    show_upload()
elif menu_option == "📈 Portfolio Overview":
    show_overview()
elif menu_option == "📉 Performance & Risk Analytics":
    show_performance()
