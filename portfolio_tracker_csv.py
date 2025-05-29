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

def calculate_cagr(start_value: float, end_value: float, periods: float) -> float:
    """Calculate annual growth rate from start to end value over given years."""
    return (end_value / start_value) ** (1 / periods) - 1


def calculate_max_drawdown(series: pd.Series) -> float:
    """Calculate worst drop from peak (risk metric)."""
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def fetch_annual_dividends(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """Fetch and aggregate dividends per year for a given ticker between two dates."""
    stock = yf.Ticker(ticker)
    dividends = stock.dividends.loc[start_date:end_date]
    return dividends.groupby(dividends.index.year).sum() if not dividends.empty else pd.Series(dtype=float)

# ---------- Streamlit app setup ----------

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# ---------- Sidebar navigation menu ----------
with st.sidebar:
    st.title("📊 Portfolio Menu")
    menu = st.radio(
        "Navigation",
        ["📁 Upload CSV", "📈 Portfolio Overview", "📉 Performance & Risk Analytics"]
    )
    if "last_updated" in st.session_state:
        st.caption(f"Last updated: {st.session_state['last_updated']}")
    if "portfolio_filename" in st.session_state:
        st.caption(f"File: {st.session_state['portfolio_filename']}")

# ---------- Initialize session state ----------
if "portfolio_file" not in st.session_state:
    st.session_state.portfolio_file = None
    st.session_state.portfolio_filename = None

# ---------- File upload section ----------
if menu == "📁 Upload CSV":
    st.title("📁 Upload Portfolio CSV")
    st.info("Upload a CSV with columns: Ticker, Shares, Buy Price, Buy Date")
    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded:
        st.session_state.portfolio_file = uploaded.read()
        st.session_state.portfolio_filename = uploaded.name
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("✅ File uploaded successfully. Use the sidebar to continue.")

# ---------- Read and parse file content if available ----------
file_content = st.session_state.get('portfolio_file')
if file_content and menu != "📁 Upload CSV":
    # Load DataFrame
    try:
        decoded = StringIO(file_content.decode('utf-8'))
        df = pd.read_csv(decoded)
        if df.empty:
            st.error("❌ Uploaded file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

    # Validate
    req = ["Ticker", "Shares", "Buy Price", "Buy Date"]
    if not all(c in df.columns for c in req):
        st.error("❌ CSV must contain: Ticker, Shares, Buy Price, Buy Date")
        st.stop()

    df["Buy Date"] = pd.to_datetime(df["Buy Date"])
    today = datetime.today()

    # Fetch data
    names, prices, divs, hist = {}, [], {}, {}
    for _, r in df.iterrows():
        t, bd = r['Ticker'], r['Buy Date']
        tk = yf.Ticker(t)
        hist[t] = tk.history(start=bd, end=today)
        names[t] = tk.info.get('shortName', t) if tk.info else t
        divs[t] = fetch_annual_dividends(t, bd, today)
        prices.append(hist[t]['Close'].iloc[-1] if not hist[t].empty else np.nan)

    df['Current'] = prices
    df['Value'] = df['Current'] * df['Shares']
    df['Invested'] = df['Shares'] * df['Buy Price']
    df['Abs Perf'] = df['Value'] - df['Invested']
    df['Rel Perf'] = df['Abs Perf'] / df['Invested']
    df['Name'] = df['Ticker'].map(names)

    total_val = df['Value'].sum()
    total_pl = df['Abs Perf'].sum()

    if menu == "📈 Portfolio Overview":
        st.title("📈 Portfolio Overview")
        # Positions listing
        st.subheader("Positions")
        sort_by = st.selectbox("Sort by", ['Rel Perf', 'Abs Perf', 'Value'])
        df = df.sort_values(by=sort_by, ascending=False)
        for _, r in df.iterrows():
            c1, c2 = st.columns([3,1])
            with c1:
                st.markdown(f"**{r['Name']}**  
<small>{r['Ticker']}</small>", unsafe_allow_html=True)
            with c2:
                st.metric("Size (€)", f"€{r['Value']:.2f}")
                st.metric("Abs (€)", f"€{r['Abs Perf']:.2f}")
                st.metric("Rel (%)", f"{r['Rel Perf']*100:.2f}%")

        # Summary metrics
        st.subheader("Overview")
        a, b = st.columns(2)
        a.metric("Total Value", f"€{total_val:.2f}")
        b.metric("Total P/L", f"€{total_pl:.2f}")

        # Pie chart
        st.subheader("Allocation")
        fig, ax = plt.subplots()
        ax.pie(df['Value'], labels=df['Ticker'], autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

        # Dividends
        st.subheader("Dividends Received")
        dv = pd.DataFrame(divs).fillna(0).sort_index()
        if not dv.empty:
            fig2, ax2 = plt.subplots()
            dv.plot(kind='bar', stacked=True, ax=ax2)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('€ Dividends')
            ax2.set_title('Annual Dividends')
            st.pyplot(fig2)
        else:
            st.info("No dividends data.")

    elif menu == "📉 Performance & Risk Analytics":
        st.title("📉 Performance & Risk Analytics")
        # unchanged
