import streamlit as st  # Import Streamlit for creating interactive web apps
import pandas as pd  # pandas for data manipulation (DataFrame structures)
import yfinance as yf  # yfinance for pulling stock market data
import matplotlib.pyplot as plt  # matplotlib for creating plots/charts
import numpy as np  # numpy for numerical operations (e.g., math functions, arrays)
from datetime import datetime, date  # datetime/date for working with dates and timestamps
from scipy.stats import linregress  # linregress from scipy to calculate linear regression (used for beta)
from io import StringIO  # StringIO lets us treat strings like file objects
from typing import Dict, Any  # Typing module for clearer type hints in functions

# ====================================
# Configuration & Constants
# ====================================
BENCHMARK_TICKER = "^GSPC"  # The S&P 500 index symbol used as a benchmark
PIE_CHART_COLORS = plt.get_cmap("tab20").colors  # Color palette for pie charts, using matplotlib's 'tab20' color map

# A dictionary containing text labels and messages for the app's user interface
TEXT = {
    "app_title": "Portfolio Analyzer",  # The title displayed in the browser tab and top of the page
    "sidebar_title": "📊 Portfolio Menu",  # The title of the sidebar menu
    "sidebar_menu": "Navigation",  # Label for the radio button in the sidebar
    "menu_upload": "📁 Upload CSV",  # Sidebar option to upload a CSV file
    "menu_overview": "📈 Portfolio Overview",  # Sidebar option for portfolio overview
    "menu_analytics": "📉 Performance & Risk Analytics",  # Sidebar option for performance/risk analysis
    "upload_csv_title": "Upload Portfolio CSV",  # Title for the CSV upload section
    "upload_csv_info": "Columns: Ticker, Shares, Buy Price, Buy Date",  # Info about required CSV columns
    "upload_button_label": "Upload CSV",  # Label for the file upload button
    "upload_success": "File uploaded successfully.",  # Success message after file upload
    "upload_error": "Unable to read CSV. Please ensure it's a valid UTF-8-encoded CSV.",  # Error message for invalid uploads
    "no_portfolio_message": "Please upload a portfolio CSV to get started.",  # Message shown if no data is loaded
    "last_updated": "Last updated",  # Label for showing the last update timestamp
    "file_name": "File",  # Label for showing the uploaded file name
    "overview_title": "Portfolio Overview",  # Title for the portfolio overview section
    "analytics_title": "Performance & Risk Analytics",  # Title for performance & risk analytics section
    "position_size_label": "Position Size (€)",  # Label for the column showing position size in EUR
    "absolute_perf_label": "Absolute Perf (€)",  # Label for absolute performance in EUR
    "relative_perf_label": "Relative Perf (%)",  # Label for relative performance in %
    "summary_subheader": "Summary",  # Subheader label in the overview/analytics sections
    "allocation_by_value_label": "Allocation by Value",  # Label for pie chart of position sizes
    "allocation_by_sector_label": "Allocation by Sector",  # Label for pie chart of sector allocation
    "portfolio_summary": "Portfolio Summary",  # Label for section summarizing portfolio metrics
    "total_return_label": "Total Return (%)",  # Label for total return
    "cagr_label": "CAGR (%)",  # Label for Compound Annual Growth Rate
    "volatility_label": "Volatility",  # Label for volatility
    "beta_label": "Beta",  # Label for beta (measure of systematic risk)
    "income_yield_label": "Income Yield (%)",  # Label for income yield
    "max_drawdown_label": "Max Drawdown",  # Label for maximum drawdown
    "sharpe_label": "Sharpe Ratio",  # Label for Sharpe ratio (risk-adjusted return)
    "sortino_label": "Sortino Ratio",  # Label for Sortino ratio (downside risk-adjusted return)
    "cumulative_return_label": "Cumulative Return",  # Label for cumulative return chart
    "received_dividends_label": "Received Dividends",  # Label for dividends received chart
    "annual_dividends_chart_title": "Annual Dividends Received",  # Label for annual dividends chart
    "no_dividends_message": "No dividends.",  # Message if no dividends available
}
def calculate_cagr(start_value: float, end_value: float, periods: float) -> float:  # Function to calculate Compound Annual Growth Rate
    if start_value <= 0 or periods <= 0:  # Check if the start value or the period is invalid (non-positive)
        return np.nan  # Return "Not a Number" to indicate an error
    return (end_value / start_value) ** (1 / periods) - 1  # Apply CAGR formula and return result


def calculate_max_drawdown(returns_series: pd.Series) -> float:  # Function to calculate the maximum drawdown
    cumulative = (1 + returns_series).cumprod()  # Compute the cumulative returns from daily returns
    peak = cumulative.cummax()  # Calculate the cumulative maximum up to each point (rolling peak)
    drawdown = (cumulative - peak) / peak  # Calculate the drawdown percentage from peak
    return drawdown.min()  # Return the maximum drawdown (worst decline)


def compute_risk_metrics(price_history: pd.DataFrame, benchmark_returns: pd.Series) -> Dict[str, float]:  # Function to calculate volatility and beta
    daily_returns = price_history["Close"].pct_change().dropna()  # Calculate daily returns from closing prices, skip missing values
    volatility = daily_returns.std() * np.sqrt(252)  # Calculate annualized volatility (assuming 252 trading days)
    paired = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()  # Combine stock and benchmark returns, remove rows with missing data
    beta = linregress(paired.iloc[:, 1], paired.iloc[:, 0])[0] if not paired.empty else np.nan  # Use linear regression to get beta (slope)
    return {"volatility": volatility, "beta": beta}  # Return both as a dictionary


@st.cache_data(ttl=3600)  # Decorator to cache function results for 1 hour to avoid re-fetching data
def get_price_history(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:  # Function to get historical price data
    try:
        stock = yf.Ticker(ticker)  # Create a yfinance Ticker object for the given symbol
        history = stock.history(start=start_date, end=end_date)  # Get the historical price data
        return history  # Return the DataFrame with the data
    except Exception:
        return pd.DataFrame()  # If there's an error, return an empty DataFrame


@st.cache_data(ttl=3600)  # Decorator to cache function results for 1 hour
def fetch_annual_dividends(ticker: str, start_date: date, end_date: date) -> pd.Series:  # Function to get annual dividend data
    try:
        stock = yf.Ticker(ticker)  # Create a Ticker object
        dividends = stock.dividends  # Get the dividend history (as a Series)
        if dividends.empty:  # If no dividend data
            return pd.Series(dtype=float)  # Return an empty series
        idx = dividends.index.tz_localize(None) if dividends.index.tz is not None else dividends.index  # Remove timezone info if present
        mask = (idx >= pd.to_datetime(start_date)) & (idx <= pd.to_datetime(end_date))  # Create a mask to filter dates
        filtered = dividends.copy()  # Make a copy of dividends
        filtered.index = idx  # Set index without timezone
        filtered = filtered.loc[mask]  # Apply the date filter
        return filtered.groupby(filtered.index.year).sum() if not filtered.empty else pd.Series(dtype=float)  # Group by year and sum
    except Exception:
        return pd.Series(dtype=float)  # If error, return empty series


def format_market_cap(value: float) -> str:  # Function to format market cap numbers for readability
    if pd.isna(value):  # If the value is NaN
        return "N/A"  # Return "Not Available"
    if value >= 1e9:  # If value is greater than or equal to 1 billion
        return f"€{value/1e9:.2f} Bn."  # Show in billions
    if value >= 1e6:  # If value is greater than or equal to 1 million
        return f"€{value/1e6:.2f} Mio."  # Show in millions
    if value >= 1e3:  # If value is greater than or equal to 1 thousand
        return f"€{value/1e3:.2f} k"  # Show in thousands
    return f"€{value:.2f}"  # Otherwise show the exact number


def validate_portfolio_df(df: pd.DataFrame) -> None:  # Function to validate the uploaded CSV data
    required_columns = ["Ticker", "Shares", "Buy Price", "Buy Date"]  # List of required columns
    missing = [col for col in required_columns if col not in df.columns]  # Check if any required columns are missing
    if missing:  # If missing columns found
        raise ValueError(f"CSV must contain: {', '.join(missing)}")  # Raise an error indicating missing columns
    if not pd.api.types.is_numeric_dtype(df["Shares"]) or (df["Shares"] < 0).any():  # Check if 'Shares' is numeric and non-negative
        raise ValueError("'Shares' must be numeric and non-negative.")  # Raise error
    if not pd.api.types.is_numeric_dtype(df["Buy Price"]) or (df["Buy Price"] < 0).any():  # Check 'Buy Price' validity
        raise ValueError("'Buy Price' must be numeric and non-negative.")  # Raise error
    try:
        pd.to_datetime(df["Buy Date"])  # Attempt to convert 'Buy Date' to datetime
    except Exception:
        raise ValueError("Column 'Buy Date' must be parsable as dates.")  # Raise error if invalid
# ====================================
# Streamlit App
# ====================================
st.set_page_config(page_title=TEXT["app_title"], layout="wide")  # Set the page title and layout (wide mode for spacious layout)

# Initialize session state for portfolio data
if "portfolio_df" not in st.session_state:  # If no portfolio data exists in session state
    st.session_state.portfolio_df = None  # Initialize as None
    st.session_state.portfolio_filename = None  # Initialize file name as None
    st.session_state.last_updated = None  # Initialize last updated timestamp as None

# Sidebar: Navigation and Captions
with st.sidebar:  # Create a sidebar for navigation
    st.title(TEXT["sidebar_title"])  # Display sidebar title
    menu_option = st.radio(  # Create a radio button selection for different app sections
        TEXT["sidebar_menu"],  # Label for the radio buttons
        [TEXT["menu_upload"], TEXT["menu_overview"], TEXT["menu_analytics"]],  # Available menu options
    )

    # Show last updated timestamp if it exists
    if st.session_state.last_updated:  # If last updated data is available
        st.caption(f"{TEXT['last_updated']}: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")  # Show formatted timestamp
    # Show uploaded file name if it exists
    if st.session_state.portfolio_filename:  # If file name is available
        st.caption(f"{TEXT['file_name']}: {st.session_state.portfolio_filename}")  # Show the file name

# ====================================
# Global CSS Overrides for layout and fonts
# ====================================
st.markdown(  # Inject custom CSS into the app for styling
    """
    <style>
    .css-10trblm { font-size: 1.2rem !important; }  /* Make metric label font larger */
    .css-1r3r1j0 { font-size: 2rem !important; }    /* Make metric value font larger */
    h2 { font-size: 1.6rem !important; margin-top: 1rem !important; margin-bottom: 0.5rem !important; }  /* Headings spacing */
    h3 { font-size: 1.4rem !important; margin-top: 1rem !important; margin-bottom: 0.3rem !important; }  /* Subheadings spacing */
    .section-spacing { margin-top: 20px; margin-bottom: 20px; }  /* Add vertical spacing between sections */
    </style>
    """,
    unsafe_allow_html=True,  # Allow HTML styling
)

# ====================================
# Upload CSV Section
# ====================================
if menu_option == TEXT["menu_upload"]:  # If the user selects the "Upload CSV" section
    st.title(TEXT["upload_csv_title"])  # Display section title
    st.info(TEXT["upload_csv_info"])  # Show information about required CSV columns
    uploaded_file = st.file_uploader(TEXT["upload_button_label"], type=["csv"])  # Create a file uploader widget for CSV files

    if uploaded_file:  # If a file has been uploaded by the user
        try:
            raw_df = pd.read_csv(uploaded_file)  # Read the CSV file into a DataFrame
            validate_portfolio_df(raw_df)  # Validate the uploaded CSV to check structure and data
            raw_df["Buy Date"] = pd.to_datetime(raw_df["Buy Date"])  # Convert 'Buy Date' column to datetime format
            st.session_state.portfolio_df = raw_df.copy()  # Save a copy of the DataFrame in session state
            st.session_state.portfolio_filename = uploaded_file.name  # Save the file name in session state
            st.session_state.last_updated = datetime.now()  # Save current timestamp
            st.success(TEXT["upload_success"])  # Show success message to the user
        except ValueError as ve:  # If a validation error occurs
            st.error(f"{TEXT['upload_error']} {ve}")  # Display the specific validation error
        except Exception as e:  # If any other error occurs
            st.error(f"{TEXT['upload_error']} {e}")  # Display a generic error message

    if st.session_state.portfolio_df is None:  # If there’s still no valid data (no file uploaded or error)
        st.stop()  # Stop execution of the app for this user session

# ====================================
# Retrieve Portfolio Data if Uploaded
# ====================================
df_portfolio = st.session_state.portfolio_df  # Load the uploaded portfolio DataFrame from session state
today_date = datetime.today().date()  # Get today’s date for later comparisons

# If no portfolio data is available but user clicked on other sections (overview or analytics), stop and inform
if df_portfolio is None and menu_option in [TEXT["menu_overview"], TEXT["menu_analytics"]]:
    st.info(TEXT["no_portfolio_message"])  # Inform user to upload portfolio first
    st.stop()  # Stop execution to avoid errors
# ====================================
# Fetch Data for Each Ticker in Portfolio
# ====================================
unique_tickers = df_portfolio["Ticker"].unique().tolist()  # Get the list of unique ticker symbols in the portfolio
analysis_start = df_portfolio["Buy Date"].min().date()  # Determine the earliest purchase date in the portfolio
analysis_end = today_date  # Set analysis end date as today's date

# Fetch historical data for the benchmark (S&P 500 index) and calculate daily returns
benchmark_history = get_price_history(BENCHMARK_TICKER, analysis_start, analysis_end)  # Get historical price data for benchmark
benchmark_returns = benchmark_history["Close"].pct_change().dropna()  # Calculate daily benchmark returns (remove NaNs)

# Initialize dictionaries to store fetched and calculated data for each ticker
names_map: Dict[str, str] = {}  # Map ticker to its name
prices_map: Dict[str, float] = {}  # Map ticker to its latest price
dividends_map: Dict[str, pd.Series] = {}  # Map ticker to its dividend history
price_histories: Dict[str, pd.DataFrame] = {}  # Map ticker to its historical price data
info_map: Dict[str, Dict[str, Any]] = {}  # Map ticker to company info (sector, etc.)
risk_metrics_map: Dict[str, Dict[str, float]] = {}  # Map ticker to risk metrics (volatility, beta)

# Loop through each ticker symbol and fetch relevant data
for idx, ticker_symbol in enumerate(unique_tickers, start=1):  # Loop with index for progress tracking
    with st.spinner(f"Fetching data for {ticker_symbol} ({idx}/{len(unique_tickers)}) …"):  # Show spinner while processing
        try:
            stock = yf.Ticker(ticker_symbol)  # Create yfinance Ticker object
            info = stock.info  # Retrieve company information
            info_map[ticker_symbol] = info  # Store info in map

            names_map[ticker_symbol] = info.get("shortName", ticker_symbol)  # Get short name of company, fallback to ticker symbol
            hist_df = get_price_history(ticker_symbol, analysis_start, analysis_end)  # Get historical price data
            price_histories[ticker_symbol] = hist_df  # Store historical price data
            prices_map[ticker_symbol] = hist_df["Close"].iloc[-1] if not hist_df.empty else np.nan  # Latest price or NaN if empty

            # Get earliest buy date for this ticker to fetch dividends from that point onward
            buy_date_min = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Buy Date"].min().date()
            dividends_map[ticker_symbol] = fetch_annual_dividends(ticker_symbol, buy_date_min, today_date)  # Fetch annual dividends

            # Calculate risk metrics (volatility and beta)
            risk_metrics_map[ticker_symbol] = compute_risk_metrics(hist_df, benchmark_returns)
        except Exception:  # In case of any error
            # Store fallback values (empty or NaN) to prevent app crashes
            names_map[ticker_symbol] = ticker_symbol
            prices_map[ticker_symbol] = np.nan
            dividends_map[ticker_symbol] = pd.Series(dtype=float)
            info_map[ticker_symbol] = {}
            risk_metrics_map[ticker_symbol] = {"volatility": np.nan, "beta": np.nan}

# ====================================
# Add Calculated Data to Portfolio DataFrame
# ====================================
df_portfolio["Current"] = df_portfolio["Ticker"].map(prices_map)  # Add latest prices to portfolio DataFrame
df_portfolio["Value"] = df_portfolio["Current"] * df_portfolio["Shares"]  # Calculate current value of each position
df_portfolio["Invested"] = df_portfolio["Buy Price"] * df_portfolio["Shares"]  # Calculate total invested amount for each position
df_portfolio["Abs Perf"] = df_portfolio["Value"] - df_portfolio["Invested"]  # Absolute performance (€ gain/loss)
df_portfolio["Rel Perf"] = df_portfolio["Abs Perf"] / df_portfolio["Invested"]  # Relative performance (%) = Abs Perf / Invested
df_portfolio["Name"] = df_portfolio["Ticker"].map(names_map)  # Add company names to DataFrame
df_portfolio["Sector"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("sector", "N/A"))  # Add sector info
df_portfolio["Industry"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("industry", "N/A"))  # Add industry info
df_portfolio["P/E"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("trailingPE", np.nan))  # Add P/E ratio
df_portfolio["Market Cap"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("marketCap", np.nan))  # Add market cap
df_portfolio["Dividend Yield (%)"] = df_portfolio["Ticker"].map(  # Calculate dividend yield as percentage
    lambda t: info_map.get(t, {}).get("dividendYield", 0.0) * 100
)
df_portfolio["P/B"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("priceToBook", np.nan))  # Add P/B ratio

# Calculate total portfolio value and total absolute profit/loss
total_portfolio_value = df_portfolio["Value"].sum()  # Sum of all current values
total_portfolio_pl = df_portfolio["Abs Perf"].sum()  # Sum of all absolute performances (P/L)
# ====================================
# Portfolio Overview Section
# ====================================
if menu_option == TEXT["menu_overview"]:  # If the user selected 'Portfolio Overview' in the sidebar
    st.title(TEXT["overview_title"])  # Display the section title
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)  # Add some vertical space

    # Prepare DataFrame for the overview table with selected columns
    overview_df = df_portfolio[["Name", "Ticker", "Value", "Abs Perf", "Rel Perf", "Sector", "Industry"]].copy()  # Select important columns
    overview_df.rename(  # Rename columns to user-friendly labels
        columns={
            "Value": TEXT["position_size_label"],
            "Abs Perf": TEXT["absolute_perf_label"],
            "Rel Perf": TEXT["relative_perf_label"],
        },
        inplace=True,
    )
    overview_df[TEXT["relative_perf_label"]] *= 100  # Convert relative performance to percentage

    # Display the DataFrame as an interactive table with formatted numbers
    st.dataframe(
        overview_df.style.format({
            TEXT["position_size_label"]: "€{:.2f}",
            TEXT["absolute_perf_label"]: "€{:.2f}",
            TEXT["relative_perf_label"]: "{:.2f}%",
        }),
        use_container_width=True,
    )

    # Display a summary of the total portfolio value and profit/loss
    st.subheader(TEXT["summary_subheader"])  # Show subheader "Summary"
    col1, col2 = st.columns(2)  # Create two columns for better layout
    col1.metric("Total Value", f"€{total_portfolio_value:.2f}")  # Show total portfolio value
    col2.metric("Total P/L", f"€{total_portfolio_pl:.2f}")  # Show total profit/loss

    # Pie chart for allocation by value
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)  # Add spacing
    st.subheader(TEXT["allocation_by_value_label"])  # Subheader for allocation by value pie chart
    fig_value, ax_value = plt.subplots(figsize=(4, 2.5))  # Create figure for pie chart
    wedges_value, texts_value, autotexts_value = ax_value.pie(
        df_portfolio["Value"],  # Pie slices = position sizes
        labels=df_portfolio["Ticker"],  # Show ticker symbols
        autopct="%1.1f%%",  # Show percentages on chart
        startangle=140,  # Rotate for better alignment
        colors=PIE_CHART_COLORS,  # Use predefined color palette
    )
    for t in texts_value: t.set_fontsize(6)  # Adjust font size for ticker labels
    for at in autotexts_value: at.set_fontsize(5)  # Adjust font size for percentages
    ax_value.axis("equal")  # Ensure pie chart is circular
    fig_value.tight_layout()  # Improve layout
    st.pyplot(fig_value)  # Render pie chart in app

    # Pie chart for allocation by sector
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.subheader(TEXT["allocation_by_sector_label"])  # Subheader for allocation by sector

    sector_alloc = df_portfolio.groupby("Sector")["Value"].sum().reset_index().sort_values("Value", ascending=False)  # Group and sort by sector
    if not sector_alloc.empty:  # Check if there is sector data
        fig_sector, ax_sector = plt.subplots(figsize=(4, 2.5))
        wedges_sector, texts_sector, autotexts_sector = ax_sector.pie(
            sector_alloc["Value"],  # Pie slices by sector
            labels=sector_alloc["Sector"],  # Sector names as labels
            autopct="%1.1f%%",
            startangle=140,
            colors=PIE_CHART_COLORS,
        )
        for t in texts_sector: t.set_fontsize(6)  # Adjust font size
        for at in autotexts_sector: at.set_fontsize(5)  # Adjust font size
        ax_sector.axis("equal")
        fig_sector.tight_layout()
        st.pyplot(fig_sector)  # Display sector allocation chart
    else:
        st.info("No sector data available.")  # If no sector data, show info message
# ====================================
# Performance & Risk Analytics Section
# ====================================
elif menu_option == TEXT["menu_analytics"]:  # If the user selected "Performance & Risk Analytics"
    st.title(TEXT["analytics_title"])  # Show section title
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)  # Add vertical space

    returns_list = []  # Create a list to store daily returns for each stock
    for ticker_symbol in unique_tickers:  # Loop through each ticker in the portfolio
        price_history = price_histories.get(ticker_symbol, pd.DataFrame())  # Retrieve historical price data
        daily_returns = (
            price_history["Close"].pct_change().dropna() if not price_history.empty else pd.Series(dtype=float)
        )  # Calculate daily returns
        returns_list.append(daily_returns)  # Add to list

        # Retrieve risk and fundamental metrics for the stock
        metrics = risk_metrics_map.get(ticker_symbol, {"volatility": np.nan, "beta": np.nan})  # Get volatility & beta
        pe_ratio = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "P/E"].iloc[0]  # P/E ratio
        market_cap_value = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Market Cap"].iloc[0]  # Market cap
        market_cap_str = format_market_cap(market_cap_value)  # Format market cap for display
        dividend_yield = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Dividend Yield (%)"].iloc[0]  # Dividend yield
        pb_ratio = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "P/B"].iloc[0]  # P/B ratio

        # Create an expandable section for detailed analytics per stock
        with st.expander(f"{names_map.get(ticker_symbol, ticker_symbol)} ({ticker_symbol})"):
            st.write(f"P/E Ratio: {pe_ratio:.2f}")  # Show P/E ratio
            st.write(f"P/B Ratio: {pb_ratio:.2f}")  # Show P/B ratio
            st.write(f"Dividend Yield: {dividend_yield:.2f}%")  # Show dividend yield
            st.write(f"Market Cap: {market_cap_str}")  # Show market cap
            st.write(f"Volatility: {metrics['volatility']:.4f}")  # Show volatility
            st.write(f"Beta: {metrics['beta']:.4f}")  # Show beta

    # Calculate portfolio-level risk/return metrics if data exists
    if returns_list:
        portfolio_returns = pd.concat(returns_list, axis=1).mean(axis=1).dropna()  # Mean daily return across all assets
        portfolio_mean = portfolio_returns.mean()  # Average daily return
        portfolio_std = portfolio_returns.std()  # Standard deviation of daily returns
        sharpe_ratio = (portfolio_mean / portfolio_std) * np.sqrt(252) if portfolio_std != 0 else np.nan  # Sharpe Ratio
        downside_std = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)  # Downside deviation for Sortino
        sortino_ratio = (portfolio_mean / downside_std) if downside_std != 0 else np.nan  # Sortino Ratio
        max_drawdown = calculate_max_drawdown(portfolio_returns)  # Max drawdown from peak

        # Calculate cumulative return and CAGR
        num_days = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days  # Days between first & last data
        num_years = num_days / 365.25 if num_days > 0 else np.nan  # Convert to years
        cumulative_return = (1 + portfolio_returns).prod()  # Final cumulative return factor
        cagr = calculate_cagr(1, cumulative_return, num_years) if num_years > 0 else np.nan  # CAGR
        total_return_pct = (cumulative_return - 1) * 100 if not np.isnan(cumulative_return) else np.nan  # Total return %

        # Portfolio-level volatility & beta vs. benchmark
        portfolio_volatility = portfolio_std * np.sqrt(252)  # Annualized volatility
        paired_portfolio = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()  # Combine w/ benchmark
        portfolio_beta = (
            linregress(paired_portfolio.iloc[:, 1], paired_portfolio.iloc[:, 0])[0] if not paired_portfolio.empty else np.nan
        )  # Beta calculation

        # Calculate income yield from dividends in the previous year
        previous_year = today_date.year - 1
        total_income_last_year = sum(
            dividends_map[ticker_symbol].get(previous_year, 0) * df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Shares"].iloc[0]
            for ticker_symbol in unique_tickers
        )
        income_yield_pct = (total_income_last_year / total_portfolio_value) * 100 if total_portfolio_value else 0.0  # Income yield %

        # Show portfolio-level summary metrics
        st.subheader(TEXT["portfolio_summary"])
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        row1_cols = st.columns([1, 1, 1, 1])  # 4-column layout for metrics
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
        