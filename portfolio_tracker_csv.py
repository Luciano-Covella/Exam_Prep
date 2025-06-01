import streamlit as st  # Streamlit for creating interactive web apps
import pandas as pd  # pandas for data manipulation and analysis
import yfinance as yf  # yfinance to fetch stock market data
import matplotlib.pyplot as plt  # matplotlib for plotting charts
import numpy as np  # numpy for numerical operations
from datetime import datetime, date  # datetime/date for working with dates
from scipy.stats import linregress  # linregress for calculating beta (linear regression)
from io import StringIO  # StringIO for treating strings as file-like objects
from typing import Dict, Any  # Typing for clearer variable type hints

# ====================================
# Configuration & Constants
# ====================================
BENCHMARK_TICKER = "^GSPC"  # S&P 500 index symbol used as a benchmark
PIE_CHART_COLORS = plt.get_cmap("tab20").colors  # Color palette for pie charts

# Text labels for the app’s interface
TEXT = {
    "app_title": "Portfolio Analyzer",  # App title
    "sidebar_title": "📊 Portfolio Menu",  # Sidebar title
    "sidebar_menu": "Navigation",  # Sidebar navigation header
    "menu_upload": "📁 Upload CSV",  # Menu item for uploading CSV
    "menu_overview": "📈 Portfolio Overview",  # Menu item for portfolio overview
    "menu_analytics": "📉 Performance & Risk Analytics",  # Menu item for analytics
    "upload_csv_title": "Upload Portfolio CSV",  # Title for upload section
    "upload_csv_info": "Columns: Ticker, Shares, Buy Price, Buy Date",  # CSV format hint
    "upload_button_label": "Upload CSV",  # Upload button label
    "upload_success": "File uploaded successfully.",  # Upload success message
    "upload_error": "Unable to read CSV. Please ensure it's a valid UTF-8-encoded CSV.",  # Upload error message
    "no_portfolio_message": "Please upload a portfolio CSV to get started.",  # Prompt if no data
    "last_updated": "Last updated",  # Label for last upload date
    "file_name": "File",  # Label for file name
    "overview_title": "Portfolio Overview",  # Portfolio overview section title
    "analytics_title": "Performance & Risk Analytics",  # Analytics section title
    "position_size_label": "Position Size (€)",  # Column label
    "absolute_perf_label": "Absolute Perf (€)",  # Column label
    "relative_perf_label": "Relative Perf (%)",  # Column label
    "summary_subheader": "Summary",  # Subheader text
    "allocation_by_value_label": "Allocation by Value",  # Allocation chart label
    "allocation_by_sector_label": "Allocation by Sector",  # Allocation chart label
    "portfolio_summary": "Portfolio Summary",  # Portfolio summary section label
    "total_return_label": "Total Return (%)",  # Metric label
    "cagr_label": "CAGR (%)",  # Metric label
    "volatility_label": "Volatility",  # Metric label
    "beta_label": "Beta",  # Metric label
    "income_yield_label": "Income Yield (%)",  # Metric label
    "max_drawdown_label": "Max Drawdown",  # Metric label
    "sharpe_label": "Sharpe Ratio",  # Metric label
    "sortino_label": "Sortino Ratio",  # Metric label
    "cumulative_return_label": "Cumulative Return",  # Chart title
    "received_dividends_label": "Received Dividends",  # Chart title
    "annual_dividends_chart_title": "Annual Dividends Received",  # Chart title
    "no_dividends_message": "No dividends.",  # Message if no dividend data
}

# ====================================
# Helper Functions
# ====================================

def calculate_cagr(start_value: float, end_value: float, periods: float) -> float:  # Calculate Compound Annual Growth Rate (CAGR)
    if start_value <= 0 or periods <= 0:  # Check for valid input
        return np.nan  # Return NaN if invalid
    return (end_value / start_value) ** (1 / periods) - 1  # CAGR formula

def calculate_max_drawdown(returns_series: pd.Series) -> float:  # Calculate maximum drawdown (largest drop from peak)
    cumulative = (1 + returns_series).cumprod()  # Cumulative returns
    peak = cumulative.cummax()  # Rolling maximum
    drawdown = (cumulative - peak) / peak  # Drawdown percentage
    return drawdown.min()  # Return worst drawdown

def compute_risk_metrics(price_history: pd.DataFrame, benchmark_returns: pd.Series) -> Dict[str, float]:  # Calculate volatility and beta
    daily_returns = price_history["Close"].pct_change().dropna()  # Daily returns
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    paired = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()  # Match data with benchmark
    beta = linregress(paired.iloc[:, 1], paired.iloc[:, 0])[0] if not paired.empty else np.nan  # Calculate beta if data exists
    return {"volatility": volatility, "beta": beta}  # Return as dictionary

@st.cache_data(ttl=3600)  # Cache for 1 hour to avoid repeated downloads
def get_price_history(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:  # Fetch historical price data
    try:
        stock = yf.Ticker(ticker)  # Create yfinance Ticker object
        history = stock.history(start=start_date, end=end_date)  # Fetch historical data
        return history  # Return data
    except Exception:
        return pd.DataFrame()  # Return empty DataFrame on error

@st.cache_data(ttl=3600)
def fetch_annual_dividends(ticker: str, start_date: date, end_date: date) -> pd.Series:  # Fetch annual dividend data
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if dividends.empty:
            return pd.Series(dtype=float)  # No dividends
        idx = dividends.index.tz_localize(None) if dividends.index.tz is not None else dividends.index  # Remove timezone
        mask = (idx >= pd.to_datetime(start_date)) & (idx <= pd.to_datetime(end_date))  # Filter dates
        filtered = dividends.copy()
        filtered.index = idx
        filtered = filtered.loc[mask]  # Apply date filter
        return filtered.groupby(filtered.index.year).sum() if not filtered.empty else pd.Series(dtype=float)  # Group by year
    except Exception:
        return pd.Series(dtype=float)  # On error, empty

def format_market_cap(value: float) -> str:  # Format market cap for readability
    if pd.isna(value): return "N/A"
    if value >= 1e9: return f"€{value/1e9:.2f} Bn."
    if value >= 1e6: return f"€{value/1e6:.2f} Mio."
    if value >= 1e3: return f"€{value/1e3:.2f} k"
    return f"€{value:.2f}"

def validate_portfolio_df(df: pd.DataFrame) -> None:  # Validate CSV data
    required_columns = ["Ticker", "Shares", "Buy Price", "Buy Date"]  # Columns needed
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain: {', '.join(missing)}")  # Error if missing columns
    if not pd.api.types.is_numeric_dtype(df["Shares"]) or (df["Shares"] < 0).any():  # Validate shares
        raise ValueError("'Shares' must be numeric and non-negative.")
    if not pd.api.types.is_numeric_dtype(df["Buy Price"]) or (df["Buy Price"] < 0).any():  # Validate price
        raise ValueError("'Buy Price' must be numeric and non-negative.")
    try:
        pd.to_datetime(df["Buy Date"])  # Check date conversion
    except Exception:
        raise ValueError("Column 'Buy Date' must be parsable as dates.")  # Error if invalid date

# ====================================
# Streamlit App
# ====================================
st.set_page_config(page_title=TEXT["app_title"], layout="wide")  # Set page title and layout (wide mode)

# Initialize session state for portfolio data and theme
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = None  # Placeholder for uploaded portfolio data
    st.session_state.portfolio_filename = None  # Placeholder for uploaded filename
    st.session_state.last_updated = None  # Placeholder for last update timestamp

if "theme" not in st.session_state:
    st.session_state.theme = "Light"  # Default to light theme

# Sidebar: Navigation, Theme Toggle, and Captions
with st.sidebar:
    st.title(TEXT["sidebar_title"])  # Sidebar title ("📊 Portfolio Menu")
    menu_option = st.radio(
        TEXT["sidebar_menu"],  # Sidebar radio label ("Navigation")
        [TEXT["menu_upload"], TEXT["menu_overview"], TEXT["menu_analytics"]],  # Options
    )

    # Theme toggle: Light or Dark
    theme_choice = st.radio(
        "Theme",  # Label
        ["Light", "Dark"],  # Choices
        index=0 if st.session_state.theme == "Light" else 1,  # Default selection
        key="theme_radio",  # Session key
        help="Toggle between Light and Dark mode",  # Help text
    )
    st.session_state.theme = theme_choice  # Save selected theme in session state

    # Show last updated timestamp if available
    if st.session_state.last_updated:
        st.caption(f"{TEXT['last_updated']}: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    # Show uploaded file name if available
    if st.session_state.portfolio_filename:
        st.caption(f"{TEXT['file_name']}: {st.session_state.portfolio_filename}")

# ====================================
# Global CSS Overrides (for layout and fonts)
# ====================================
st.markdown(
    """
    <style>
    .css-10trblm { font-size: 1.2rem !important; }  /* Metric label font */
    .css-1r3r1j0 { font-size: 2rem !important; }    /* Metric value font */
    h2 { font-size: 1.6rem !important; margin-top: 1rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 1.4rem !important; margin-top: 1rem !important; margin-bottom: 0.3rem !important; }
    .section-spacing { margin-top: 20px; margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True,  # Allow HTML for styling
)

# ====================================
# Upload CSV Section
# ====================================
if menu_option == TEXT["menu_upload"]:  # If user selected "Upload CSV" in sidebar
    st.title(TEXT["upload_csv_title"])  # Show title for upload section
    st.info(TEXT["upload_csv_info"])  # Show info about required CSV columns
    uploaded_file = st.file_uploader(TEXT["upload_button_label"], type=["csv"])  # File uploader widget

    if uploaded_file:  # If a file has been uploaded
        try:
            raw_df = pd.read_csv(uploaded_file)  # Read CSV file into DataFrame
            validate_portfolio_df(raw_df)  # Validate the CSV data for required structure
            raw_df["Buy Date"] = pd.to_datetime(raw_df["Buy Date"])  # Convert 'Buy Date' to datetime
            st.session_state.portfolio_df = raw_df.copy()  # Save a copy of DataFrame in session state
            st.session_state.portfolio_filename = uploaded_file.name  # Save the file name
            st.session_state.last_updated = datetime.now()  # Save current timestamp
            st.success(TEXT["upload_success"])  # Show success message
        except ValueError as ve:
            st.error(f"{TEXT['upload_error']} {ve}")  # Show validation error
        except Exception as e:
            st.error(f"{TEXT['upload_error']} {e}")  # Show generic error

    if st.session_state.portfolio_df is None:  # If no portfolio data available
        st.stop()  # Stop execution of the rest of the app

# ====================================
# Retrieve portfolio data (if uploaded)
# ====================================
df_portfolio = st.session_state.portfolio_df  # Load portfolio DataFrame from session state
today_date = datetime.today().date()  # Current date

# If no data and user selected overview or analytics, show info and stop
if df_portfolio is None and menu_option in [TEXT["menu_overview"], TEXT["menu_analytics"]]:
    st.info(TEXT["no_portfolio_message"])
    st.stop()

# ====================================
# Fetch Data for Each Ticker in Portfolio
# ====================================
unique_tickers = df_portfolio["Ticker"].unique().tolist()  # Get unique tickers from portfolio
analysis_start = df_portfolio["Buy Date"].min().date()  # Find earliest purchase date
analysis_end = today_date  # Set analysis end date to today

# Fetch historical data for benchmark (S&P 500) and calculate daily returns
benchmark_history = get_price_history(BENCHMARK_TICKER, analysis_start, analysis_end)
benchmark_returns = benchmark_history["Close"].pct_change().dropna()  # Daily benchmark returns

# Initialize dictionaries for storing fetched data
names_map: Dict[str, str] = {}  # Map ticker symbol to company name
prices_map: Dict[str, float] = {}  # Map ticker to current price
dividends_map: Dict[str, pd.Series] = {}  # Map ticker to dividend history
price_histories: Dict[str, pd.DataFrame] = {}  # Map ticker to historical price data
info_map: Dict[str, Dict[str, Any]] = {}  # Map ticker to additional info (sector, industry, etc.)
risk_metrics_map: Dict[str, Dict[str, float]] = {}  # Map ticker to risk metrics

# Loop through each ticker and fetch relevant data
for idx, ticker_symbol in enumerate(unique_tickers, start=1):
    with st.spinner(f"Fetching data for {ticker_symbol} ({idx}/{len(unique_tickers)}) …"):  # Show spinner while fetching
        try:
            stock = yf.Ticker(ticker_symbol)  # Create ticker object
            info = stock.info  # Fetch stock information
            info_map[ticker_symbol] = info  # Store info

            names_map[ticker_symbol] = info.get("shortName", ticker_symbol)  # Get company name
            hist_df = get_price_history(ticker_symbol, analysis_start, analysis_end)  # Get historical prices
            price_histories[ticker_symbol] = hist_df  # Store historical data
            prices_map[ticker_symbol] = hist_df["Close"].iloc[-1] if not hist_df.empty else np.nan  # Get latest price

            buy_date_min = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Buy Date"].min().date()  # Get earliest buy date
            dividends_map[ticker_symbol] = fetch_annual_dividends(ticker_symbol, buy_date_min, today_date)  # Fetch dividends

            risk_metrics_map[ticker_symbol] = compute_risk_metrics(hist_df, benchmark_returns)  # Calculate risk metrics
        except Exception:
            # If an error occurs, store fallback values
            names_map[ticker_symbol] = ticker_symbol
            prices_map[ticker_symbol] = np.nan
            dividends_map[ticker_symbol] = pd.Series(dtype=float)
            info_map[ticker_symbol] = {}
            risk_metrics_map[ticker_symbol] = {"volatility": np.nan, "beta": np.nan}

# ====================================
# Add Calculated Data to Portfolio DataFrame
# ====================================
df_portfolio["Current"] = df_portfolio["Ticker"].map(prices_map)  # Add current prices
df_portfolio["Value"] = df_portfolio["Current"] * df_portfolio["Shares"]  # Calculate position size
df_portfolio["Invested"] = df_portfolio["Buy Price"] * df_portfolio["Shares"]  # Calculate total invested
df_portfolio["Abs Perf"] = df_portfolio["Value"] - df_portfolio["Invested"]  # Calculate absolute performance (€)
df_portfolio["Rel Perf"] = df_portfolio["Abs Perf"] / df_portfolio["Invested"]  # Calculate relative performance (%)
df_portfolio["Name"] = df_portfolio["Ticker"].map(names_map)  # Add company names
df_portfolio["Sector"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("sector", "N/A"))  # Add sector
df_portfolio["Industry"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("industry", "N/A"))  # Add industry
df_portfolio["P/E"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("trailingPE", np.nan))  # Add P/E ratio
df_portfolio["Market Cap"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("marketCap", np.nan))  # Add market cap
df_portfolio["Dividend Yield (%)"] = df_portfolio["Ticker"].map(
    lambda t: info_map.get(t, {}).get("dividendYield", 0.0) * 100
)  # Add dividend yield (%)
df_portfolio["P/B"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("priceToBook", np.nan))  # Add P/B ratio

# Calculate total portfolio value and total P/L
total_portfolio_value = df_portfolio["Value"].sum()  # Total value
total_portfolio_pl = df_portfolio["Abs Perf"].sum()  # Total absolute performance

# ====================================
# Portfolio Overview Section
# ====================================
if menu_option == TEXT["menu_overview"]:  # If user selected 'Portfolio Overview' in the sidebar
    st.title(TEXT["overview_title"])  # Display the section title
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)  # Add vertical spacing

    # Prepare DataFrame for overview table with relevant columns
    overview_df = df_portfolio[["Name", "Ticker", "Value", "Abs Perf", "Rel Perf", "Sector", "Industry"]].copy()
    overview_df.rename(
        columns={
            "Value": TEXT["position_size_label"],  # Rename columns for clarity
            "Abs Perf": TEXT["absolute_perf_label"],
            "Rel Perf": TEXT["relative_perf_label"],
            "Sector": "Sector",
            "Industry": "Industry",
        },
        inplace=True,
    )
    overview_df[TEXT["relative_perf_label"]] *= 100  # Convert relative performance to percentage

    # Display DataFrame as a table with formatted numbers
    st.dataframe(
        overview_df.style.format({
            TEXT["position_size_label"]: "€{:.2f}",
            TEXT["absolute_perf_label"]: "€{:.2f}",
            TEXT["relative_perf_label"]: "{:.2f}%",
        }),
        use_container_width=True,
    )

    # Display a summary of total value and total P/L
    st.subheader(TEXT["summary_subheader"])  # Subheader: "Summary"
    col1, col2 = st.columns(2)  # Create two columns side by side
    col1.metric("Total Value", f"€{total_portfolio_value:.2f}")  # Show total portfolio value
    col2.metric("Total P/L", f"€{total_portfolio_pl:.2f}")  # Show total profit/loss

    # Pie chart for allocation by value
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.subheader(TEXT["allocation_by_value_label"])  # Subheader for allocation by value

    fig_value, ax_value = plt.subplots(figsize=(4, 2.5))  # Create a small pie chart
    wedges_value, texts_value, autotexts_value = ax_value.pie(
        df_portfolio["Value"],  # Pie slices = position sizes
        labels=df_portfolio["Ticker"],  # Label slices with ticker symbols
        autopct="%1.1f%%",  # Show percentage
        startangle=140,  # Rotate for better view
        colors=PIE_CHART_COLORS,  # Use predefined color palette
    )
    for t in texts_value: t.set_fontsize(6)  # Adjust font size for slice labels
    for at in autotexts_value: at.set_fontsize(5)  # Adjust font size for percentages
    ax_value.axis("equal")  # Make sure pie is circular
    fig_value.tight_layout()  # Adjust layout
    st.pyplot(fig_value)  # Display pie chart

    # Pie chart for allocation by sector
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.subheader(TEXT["allocation_by_sector_label"])  # Subheader for sector allocation

    # Group by sector and sum values
    sector_alloc = df_portfolio.groupby("Sector")["Value"].sum().reset_index().sort_values("Value", ascending=False)
    if not sector_alloc.empty:  # If there is sector data
        fig_sector, ax_sector = plt.subplots(figsize=(4, 2.5))
        wedges_sector, texts_sector, autotexts_sector = ax_sector.pie(
            sector_alloc["Value"],
            labels=sector_alloc["Sector"],
            autopct="%1.1f%%",
            startangle=140,
            colors=PIE_CHART_COLORS,
        )
        for t in texts_sector: t.set_fontsize(6)  # Adjust sector labels
        for at in autotexts_sector: at.set_fontsize(5)  # Adjust percentages
        ax_sector.axis("equal")
        fig_sector.tight_layout()
        st.pyplot(fig_sector)  # Show sector allocation pie chart
    else:
        st.info("No sector data available.")  # If no data, show message

# ====================================
# Performance & Risk Analytics Section
# ====================================
elif menu_option == TEXT["menu_analytics"]:  # If user selects "Performance & Risk Analytics"
    st.title(TEXT["analytics_title"])  # Show section title
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)  # Add vertical spacing

    returns_list = []  # List to store returns for each stock
    for ticker_symbol in unique_tickers:  # Loop through each ticker
        price_history = price_histories.get(ticker_symbol, pd.DataFrame())  # Get historical prices
        daily_returns = (
            price_history["Close"].pct_change().dropna() if not price_history.empty else pd.Series(dtype=float)
        )  # Calculate daily returns
        returns_list.append(daily_returns)  # Add to list

        # Retrieve metrics and fundamental data
        metrics = risk_metrics_map.get(ticker_symbol, {"volatility": np.nan, "beta": np.nan})
        pe_ratio = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "P/E"].iloc[0]
        market_cap_value = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Market Cap"].iloc[0]
        market_cap_str = format_market_cap(market_cap_value)  # Format market cap
        dividend_yield = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Dividend Yield (%)"].iloc[0]
        pb_ratio = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "P/B"].iloc[0]

        # Create expandable section for each stock's analytics
        with st.expander(f"{names_map.get(ticker_symbol, ticker_symbol)} ({ticker_symbol})"):
            st.write(f"P/E Ratio: {pe_ratio:.2f}")
            st.write(f"P/B Ratio: {pb_ratio:.2f}")
            st.write(f"Dividend Yield: {dividend_yield:.2f}%")
            st.write(f"Market Cap: {market_cap_str}")
            st.write(f"Volatility: {metrics['volatility']:.4f}")
            st.write(f"Beta: {metrics['beta']:.4f}")

    # Calculate portfolio-level returns and risk metrics
    if returns_list:
        portfolio_returns = pd.concat(returns_list, axis=1).mean(axis=1).dropna()  # Average returns across all stocks
        portfolio_mean = portfolio_returns.mean()  # Mean daily return
        portfolio_std = portfolio_returns.std()  # Standard deviation
        sharpe_ratio = (portfolio_mean / portfolio_std) * np.sqrt(252) if portfolio_std != 0 else np.nan  # Sharpe ratio
        downside_std = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)  # Downside deviation
        sortino_ratio = (portfolio_mean / downside_std) if downside_std != 0 else np.nan  # Sortino ratio
        max_drawdown = calculate_max_drawdown(portfolio_returns)  # Max drawdown

        num_days = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days  # Number of days in data
        num_years = num_days / 365.25 if num_days > 0 else np.nan  # Convert to years
        cumulative_return = (1 + portfolio_returns).prod()  # Cumulative return factor
        cagr = calculate_cagr(1, cumulative_return, num_years) if num_years > 0 else np.nan  # CAGR
        total_return_pct = (cumulative_return - 1) * 100 if not np.isnan(cumulative_return) else np.nan  # % total return

        portfolio_volatility = portfolio_std * np.sqrt(252)  # Annualized volatility
        paired_portfolio = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()  # Combine with benchmark
        portfolio_beta = (
            linregress(paired_portfolio.iloc[:, 1], paired_portfolio.iloc[:, 0])[0] if not paired_portfolio.empty else np.nan
        )  # Calculate portfolio beta

        # Calculate income yield from dividends
        previous_year = today_date.year - 1
        total_income_last_year = sum(
            dividends_map[ticker_symbol].get(previous_year, 0) * df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Shares"].iloc[0]
            for ticker_symbol in unique_tickers
        )
        income_yield_pct = (total_income_last_year / total_portfolio_value) * 100 if total_portfolio_value else 0.0

        # Display portfolio summary metrics
        st.subheader(TEXT["portfolio_summary"])
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        row1_cols = st.columns([1, 1, 1, 1])  # Create 4 equal-width columns for metrics
        row1_cols[0].metric(TEXT["total_return_label"], f"{total_return_pct:.2f}%")
        row1_cols[1].metric(TEXT["cagr_label"], f"{cagr * 100:.2f}%" if not np.isnan(cagr) else "N/A")
        row1_cols[2].metric(TEXT["volatility_label"], f"{portfolio_volatility:.2f}")
        row1_cols[3].metric(TEXT["beta_label"], f"{portfolio_beta:.2f}")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        row2_cols = st.columns([1, 1, 1, 1])
        row2_cols[0].metric(TEXT["income_yield_label"], f"{income_yield_pct:.2f}%")
        row2_cols[1].metric(TEXT["max_drawdown_label"], f"{max_drawdown:.2f}")
        row2_cols[2].metric(TEXT["sharpe_label"], f"{sharpe_ratio:.2f}")
        row2_cols[3].metric(TEXT["sortino_label"], f"{sortino_ratio:.2f}")

        # Plot cumulative returns vs. S&P 500 and other assets
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.subheader(TEXT["cumulative_return_label"])
        fig_cum, ax_cum = plt.subplots(figsize=(4, 2.5))
        ax_cum.plot((1 + portfolio_returns).cumprod(), label="Portfolio", linewidth=2, color="#1f77b4")
        sp_returns = benchmark_returns
        ax_cum.plot((1 + sp_returns).cumprod(), linestyle="--", label="S&P 500", color="#ff7f0e")
        gold_hist = get_price_history("GLD", analysis_start, analysis_end)
        gold_returns = gold_hist["Close"].pct_change().dropna()
        ax_cum.plot((1 + gold_returns).cumprod(), linestyle="--", label="Gold (GLD)", color="#2ca02c")
        btc_hist = get_price_history("BTC-USD", analysis_start, analysis_end)
        btc_returns = btc_hist["Close"].pct_change().dropna()
        ax_cum.plot((1 + btc_returns).cumprod(), linestyle="--", label="Bitcoin (BTC-USD)", color="#d62728")
        ax_cum.set_xlabel("Date", fontsize=6)
        ax_cum.set_ylabel("Cum. Return", fontsize=6)
        ax_cum.tick_params(axis="x", labelsize=5, rotation=45)
        ax_cum.tick_params(axis="y", labelsize=5)
        ax_cum.spines["top"].set_visible(False)
        ax_cum.spines["right"].set_visible(False)
        ax_cum.legend(fontsize=6, loc="upper left")
        fig_cum.tight_layout()
        st.pyplot(fig_cum)

        # Plot annual dividends received
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.subheader(TEXT["received_dividends_label"])
        dividends_adjusted: Dict[str, pd.Series] = {}
        for ticker_symbol, dividends_series in dividends_map.items():
            shares_count = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Shares"].iloc[0]
            dividends_adjusted[ticker_symbol] = dividends_series * shares_count
        dividends_df = pd.DataFrame(dividends_adjusted).fillna(0).sort_index()
        if not dividends_df.empty:
            fig_div, ax_div = plt.subplots(figsize=(4, 2.5))
            dividends_df.plot(kind="bar", stacked=True, ax=ax_div, color=PIE_CHART_COLORS)
            ax_div.set_xlabel("Year", fontsize=6)
            ax_div.set_ylabel("Dividends (€)", fontsize=6)
            ax_div.tick_params(axis="x", labelsize=5, rotation=45)
            ax_div.tick_params(axis="y", labelsize=5)
            ax_div.spines["top"].set_visible(False)
            ax_div.spines["right"].set_visible(False)
            legend = ax_div.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
            fig_div.subplots_adjust(right=0.7)
            fig_div.tight_layout()
            st.pyplot(fig_div)
        else:
            st.info(TEXT["no_dividends_message"])

        # Plot return correlation matrix
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.subheader("Return Correlation Matrix")
        all_returns_df = pd.DataFrame({
            ticker_symbol: price_histories[ticker_symbol]["Close"].pct_change()
            for ticker_symbol in unique_tickers
            if not price_histories[ticker_symbol].empty
        })
        corr_matrix = all_returns_df.corr().fillna(0)
        fig_corr, ax_corr = plt.subplots(figsize=(5, 3))
        cax = ax_corr.matshow(corr_matrix, cmap="viridis")
        fig_corr.colorbar(cax, fraction=0.046, pad=0.04)
        ax_corr.set_xticks(range(len(corr_matrix.columns)))
        ax_corr.set_yticks(range(len(corr_matrix.index)))
        ax_corr.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=5)
        ax_corr.set_yticklabels(corr_matrix.index, fontsize=5)
        ax_corr.tick_params(length=0)
        ax_corr.spines["top"].set_visible(False)
        ax_corr.spines["right"].set_visible(False)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
    else:
        st.info("Not enough data for performance & risk analytics.")  # If no returns data, show message