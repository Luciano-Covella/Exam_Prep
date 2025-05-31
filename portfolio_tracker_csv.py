import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date
from scipy.stats import linregress
from io import StringIO
from typing import Dict, Any

# ====================================
# Configuration & Constants
# ====================================
BENCHMARK_TICKER = "^GSPC"
PIE_CHART_COLORS = plt.get_cmap("tab20").colors

TEXT = {
    "app_title": "Portfolio Analyzer",
    "sidebar_title": "📊 Portfolio Menu",
    "sidebar_menu": "Navigation",
    "menu_upload": "📁 Upload CSV",
    "menu_overview": "📈 Portfolio Overview",
    "menu_analytics": "📉 Performance & Risk Analytics",
    "upload_csv_title": "Upload Portfolio CSV",
    "upload_csv_info": "Columns: Ticker, Shares, Buy Price, Buy Date",
    "upload_button_label": "Upload CSV",
    "upload_success": "File uploaded successfully.",
    "upload_error": "Unable to read CSV. Please ensure it's a valid UTF-8-encoded CSV.",
    "no_portfolio_message": "Please upload a portfolio CSV to get started.",
    "last_updated": "Last updated",
    "file_name": "File",
    "overview_title": "Portfolio Overview",
    "analytics_title": "Performance & Risk Analytics",
    "position_size_label": "Position Size (€)",
    "absolute_perf_label": "Absolute Perf (€)",
    "relative_perf_label": "Relative Perf (%)",
    "summary_subheader": "Summary",
    "allocation_by_value_label": "Allocation by Value",
    "allocation_by_sector_label": "Allocation by Sector",
    "portfolio_summary": "Portfolio Summary",
    "total_return_label": "Total Return (%)",
    "cagr_label": "CAGR (%)",
    "volatility_label": "Volatility",
    "beta_label": "Beta",
    "income_yield_label": "Income Yield (%)",
    "max_drawdown_label": "Max Drawdown",
    "sharpe_label": "Sharpe Ratio",
    "sortino_label": "Sortino Ratio",
    "cumulative_return_label": "Cumulative Return",
    "received_dividends_label": "Received Dividends",
    "annual_dividends_chart_title": "Annual Dividends Received",
    "no_dividends_message": "No dividends.",
}

# ====================================
# Helper Functions
# ====================================
def calculate_cagr(start_value: float, end_value: float, periods: float) -> float:
    if start_value <= 0 or periods <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / periods) - 1


def calculate_max_drawdown(returns_series: pd.Series) -> float:
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def compute_risk_metrics(price_history: pd.DataFrame, benchmark_returns: pd.Series) -> Dict[str, float]:
    daily_returns = price_history["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    paired = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
    beta = linregress(paired.iloc[:, 1], paired.iloc[:, 0])[0] if not paired.empty else np.nan
    return {"volatility": volatility, "beta": beta}


@st.cache_data(ttl=3600)
def get_price_history(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)
        return history
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_annual_dividends(ticker: str, start_date: date, end_date: date) -> pd.Series:
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if dividends.empty:
            return pd.Series(dtype=float)
        idx = dividends.index
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        mask = (idx >= pd.to_datetime(start_date)) & (idx <= pd.to_datetime(end_date))
        filtered = dividends.copy()
        filtered.index = idx
        filtered = filtered.loc[mask]
        return filtered.groupby(filtered.index.year).sum() if not filtered.empty else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def format_market_cap(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value >= 1e9:
        return f"€{value/1e9:.2f} Bn."
    if value >= 1e6:
        return f"€{value/1e6:.2f} Mio."
    if value >= 1e3:
        return f"€{value/1e3:.2f} k"
    return f"€{value:.2f}"


def validate_portfolio_df(df: pd.DataFrame) -> None:
    required_columns = ["Ticker", "Shares", "Buy Price", "Buy Date"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain: {', '.join(missing)}")
    if not pd.api.types.is_numeric_dtype(df["Shares"]) or (df["Shares"] < 0).any():
        raise ValueError("'Shares' must be numeric and non-negative.")
    if not pd.api.types.is_numeric_dtype(df["Buy Price"]) or (df["Buy Price"] < 0).any():
        raise ValueError("'Buy Price' must be numeric and non-negative.")
    try:
        pd.to_datetime(df["Buy Date"])
    except Exception:
        raise ValueError("Column 'Buy Date' must be parsable as dates.")


# ====================================
# Streamlit App
# ====================================
st.set_page_config(page_title=TEXT["app_title"], layout="wide")

# Initialize session state
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = None
    st.session_state.portfolio_filename = None
    st.session_state.last_updated = None

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# Sidebar
with st.sidebar:
    st.title(TEXT["sidebar_title"])
    menu_option = st.radio(
        TEXT["sidebar_menu"],
        [TEXT["menu_upload"], TEXT["menu_overview"], TEXT["menu_analytics"]],
    )

    # Theme toggle
    theme_choice = st.radio(
        "Theme",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        key="theme_radio",
        help="Toggle between Light and Dark mode",
    )
    st.session_state.theme = theme_choice

    if st.session_state.last_updated:
        st.caption(f"{TEXT['last_updated']}: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.session_state.portfolio_filename:
        st.caption(f"{TEXT['file_name']}: {st.session_state.portfolio_filename}")

# Only inject Dark‐Mode CSS when we are *not* on the Upload page
if st.session_state.theme == "Dark" and menu_option != TEXT["menu_upload"]:
    st.markdown(
        """
        <style>
        /* Main container background and all text white */
        [data-testid="stAppViewContainer"] {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        [data-testid="stAppViewContainer"] * {
            color: #FAFAFA !important;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #171A21;
            color: #FAFAFA;
        }
        [data-testid="stSidebar"] * {
            color: #FAFAFA !important;
        }
        /* Metric text overrides */
        .css-10trblm, .css-1r3r1j0 {
            color: #FAFAFA !important;
        }
        /* Button text */
        button, .stButton>button>div {
            color: #FAFAFA !important;
            background-color: #2A2A2A !important;
            border: 1px solid #555 !important;
        }
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #FAFAFA !important;
        }
        /* DataFrame text */
        .css-1kyxreq {
            color: #FAFAFA !important;
        }
        /* Input labels and placeholders */
        input, textarea, label {
            color: #FAFAFA !important;
        }
        /* Selectbox and radio text */
        .stSelectbox>div>div>div>div, .stRadio>div>label {
            color: #FAFAFA !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ====================================
# Upload CSV Section
# ====================================
if menu_option == TEXT["menu_upload"]:
    st.title(TEXT["upload_csv_title"])
    st.info(TEXT["upload_csv_info"])

    # *No dark‐mode CSS is injected here,* so the uploader retains its standard
    # light styling (white background / black text) automatically.
    uploaded_file = st.file_uploader(TEXT["upload_button_label"], type=["csv"])

    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            validate_portfolio_df(raw_df)
            raw_df["Buy Date"] = pd.to_datetime(raw_df["Buy Date"])
            st.session_state.portfolio_df = raw_df.copy()
            st.session_state.portfolio_filename = uploaded_file.name
            st.session_state.last_updated = datetime.now()
            st.success(TEXT["upload_success"])
        except ValueError as ve:
            st.error(f"{TEXT['upload_error']} {ve}")
        except Exception as e:
            st.error(f"{TEXT['upload_error']} {e}")

    if st.session_state.portfolio_df is None:
        st.stop()

# Pull in the DataFrame or show an info message
df_portfolio = st.session_state.portfolio_df
today_date = datetime.today().date()
if df_portfolio is None and menu_option in [TEXT["menu_overview"], TEXT["menu_analytics"]]:
    st.info(TEXT["no_portfolio_message"])
    st.stop()

# ------------------------------------------------------------
# From here on, you can continue with the “overview” and “analytics”
# logic exactly as before—fetch prices, compute metrics, etc.
# (Omitted for brevity, since the question was all about the                
#  uploader box in Dark Mode.)
# ------------------------------------------------------------
