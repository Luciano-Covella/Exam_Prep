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
    "upload_error": "Unable to read CSV. Please ensure it's a valid UTF-8‐encoded CSV.",
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
    "benchmarks_label": "Benchmarks",
    "custom_tickers_label": "Custom tickers (comma-separated)",
    "custom_tickers_help": "Enter additional tickers to compare (e.g., MSFT, AMZN)",
    "received_dividends_label": "Received Dividends",
    "annual_dividends_chart_title": "Annual Dividends Received",
    "no_dividends_message": "No dividends.",
    "fetching_data": "Fetching data for {ticker} ({idx}/{total}) …",
    "start_date_label": "Start Date",
    "end_date_label": "End Date",
    "show_corr_matrix_label": "Show Correlation Matrix",
}

# ====================================
# Helper Functions
# ====================================

def calculate_cagr(start_value: float, end_value: float, periods: float) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR).

    Args:
        start_value (float): The initial value.
        end_value (float): The ending value.
        periods (float): Number of years.

    Returns:
        float: CAGR as a decimal.
    """
    if start_value <= 0 or periods <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / periods) - 1


def calculate_max_drawdown(returns_series: pd.Series) -> float:
    """
    Calculate the maximum drawdown of a returns series.

    Args:
        returns_series (pd.Series): Time series of periodic returns.

    Returns:
        float: Maximum drawdown (negative value).
    """
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def compute_risk_metrics(
    price_history: pd.DataFrame, benchmark_returns: pd.Series
) -> Dict[str, float]:
    """
    Compute volatility and beta for a given price history versus a benchmark.

    Args:
        price_history (pd.DataFrame): DataFrame containing 'Close' prices indexed by date.
        benchmark_returns (pd.Series): Daily returns of the benchmark indexed by date.

    Returns:
        Dict[str, float]: Dictionary with 'volatility' and 'beta'.
    """
    daily_returns = price_history["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)

    paired = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
    if not paired.empty:
        beta = linregress(paired.iloc[:, 1], paired.iloc[:, 0])[0]
    else:
        beta = np.nan

    return {"volatility": volatility, "beta": beta}


@st.cache_data(ttl=3600)
def get_price_history(
    ticker: str, start_date: date, end_date: date
) -> pd.DataFrame:
    """
    Fetch historical price data for a ticker using yfinance, cached for one hour.

    Args:
        ticker (str): Ticker symbol.
        start_date (date): Start date for history.
        end_date (date): End date for history.

    Returns:
        pd.DataFrame: Historical price DataFrame with at least 'Close' column.
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)
        return history
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_annual_dividends(
    ticker: str, start_date: date, end_date: date
) -> pd.Series:
    """
    Fetch and sum up dividends by calendar year for a ticker between start_date and end_date.

    Args:
        ticker (str): Ticker symbol.
        start_date (date): Earliest date to include dividends.
        end_date (date): Latest date to include dividends.

    Returns:
        pd.Series: Series indexed by year with total dividends for that year.
    """
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if dividends.empty:
            return pd.Series(dtype=float)

        # Ensure index is timezone-naive
        idx = dividends.index
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        mask = (idx >= pd.to_datetime(start_date)) & (idx <= pd.to_datetime(end_date))
        filtered = dividends.copy()
        filtered.index = idx
        filtered = filtered.loc[mask]
        if not filtered.empty:
            return filtered.groupby(filtered.index.year).sum()
        else:
            return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def format_market_cap(value: float) -> str:
    """
    Format market capitalization into human-readable string with k, Mio., Bn.

    Args:
        value (float): Market cap value in base currency.

    Returns:
        str: Formatted string, e.g. "€1.23 Bn."
    """
    if pd.isna(value):
        return "N/A"
    if value >= 1e9:
        return f"€{value / 1e9:.2f} Bn."
    if value >= 1e6:
        return f"€{value / 1e6:.2f} Mio."
    if value >= 1e3:
        return f"€{value / 1e3:.2f} k"
    return f"€{value:.2f}"


def validate_portfolio_df(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains the required columns with correct types.

    Args:
        df (pd.DataFrame): The portfolio DataFrame.

    Raises:
        ValueError: If validation fails.
    """
    required_columns = ["Ticker", "Shares", "Buy Price", "Buy Date"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain: {', '.join(missing)}")
    # Check for non-negative numeric columns
    if not pd.api.types.is_numeric_dtype(df["Shares"]):
        raise ValueError("Column 'Shares' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["Buy Price"]):
        raise ValueError("Column 'Buy Price' must be numeric.")
    if (df["Shares"] < 0).any():
        raise ValueError("'Shares' must be non-negative.")
    if (df["Buy Price"] < 0).any():
        raise ValueError("'Buy Price' must be non-negative.")
    # Check date parsing
    try:
        pd.to_datetime(df["Buy Date"])
    except Exception:
        raise ValueError("Column 'Buy Date' must be parsable as dates.")


# ====================================
# Streamlit App
# ====================================
st.set_page_config(page_title=TEXT["app_title"], layout="wide")

# Initialize session state for portfolio DataFrame
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = None
    st.session_state.portfolio_filename = None
    st.session_state.last_updated = None

# Sidebar: Navigation and Captions
with st.sidebar:
    st.title(TEXT["sidebar_title"])
    menu_option = st.radio(
        TEXT["sidebar_menu"],
        [TEXT["menu_upload"], TEXT["menu_overview"], TEXT["menu_analytics"]],
    )

    if st.session_state.last_updated:
        st.caption(f"{TEXT['last_updated']}: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.session_state.portfolio_filename:
        st.caption(f"{TEXT['file_name']}: {st.session_state.portfolio_filename}")

# ===============================
# Upload CSV Section (Main Area)
# ===============================
if menu_option == TEXT["menu_upload"]:
    st.title(TEXT["upload_csv_title"])
    st.info(TEXT["upload_csv_info"])
    uploaded_file = st.file_uploader(TEXT["upload_button_label"], type=["csv"])
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            validate_portfolio_df(raw_df)
            # Parse dates
            raw_df["Buy Date"] = pd.to_datetime(raw_df["Buy Date"])
            # Store parsed DataFrame in session state
            st.session_state.portfolio_df = raw_df.copy()
            st.session_state.portfolio_filename = uploaded_file.name
            st.session_state.last_updated = datetime.now()
            st.success(TEXT["upload_success"])
        except ValueError as ve:
            st.error(f"{TEXT['upload_error']} {ve}")
        except Exception as e:
            st.error(f"{TEXT['upload_error']} {e}")

    # If no portfolio yet, stop here
    if st.session_state.portfolio_df is None:
        st.stop()

# Retrieve the validated DataFrame
df_portfolio = st.session_state.portfolio_df.copy()
today_date = datetime.today().date()

# If user hasn't uploaded and is on Overview/Analytics, prompt them
if df_portfolio is None and menu_option in [TEXT["menu_overview"], TEXT["menu_analytics"]]:
    st.info(TEXT["no_portfolio_message"])
    st.stop()

# Fetch name, price, dividends, and risk metrics for each unique ticker
unique_tickers = df_portfolio["Ticker"].unique().tolist()

# Determine date range for analytics
if menu_option == TEXT["menu_analytics"]:
    min_buy_date = df_portfolio["Buy Date"].min().date()
    start_date_input = st.date_input(
        TEXT["start_date_label"], min_value=min_buy_date, max_value=today_date, value=min_buy_date
    )
    end_date_input = st.date_input(
        TEXT["end_date_label"], min_value=start_date_input, max_value=today_date, value=today_date
    )
    analysis_start = start_date_input
    analysis_end = end_date_input
else:
    analysis_start = df_portfolio["Buy Date"].min().date()
    analysis_end = today_date

# Fetch benchmark returns once
benchmark_history = get_price_history(BENCHMARK_TICKER, analysis_start, analysis_end)
benchmark_returns = benchmark_history["Close"].pct_change().dropna()

# Initialize containers
names_map: Dict[str, str] = {}
prices_map: Dict[str, float] = {}
dividends_map: Dict[str, pd.Series] = {}
price_histories: Dict[str, pd.DataFrame] = {}
info_map: Dict[str, Dict[str, Any]] = {}
risk_metrics_map: Dict[str, Dict[str, float]] = {}

# Fetch data for each ticker
for idx, ticker_symbol in enumerate(unique_tickers, start=1):
    with st.spinner(TEXT["fetching_data"].format(ticker=ticker_symbol, idx=idx, total=len(unique_tickers))):
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            info_map[ticker_symbol] = info

            # Name
            names_map[ticker_symbol] = info.get("shortName", ticker_symbol)

            # Price history
            hist_df = get_price_history(ticker_symbol, analysis_start, analysis_end)
            price_histories[ticker_symbol] = hist_df
            if not hist_df.empty:
                prices_map[ticker_symbol] = hist_df["Close"].iloc[-1]
            else:
                prices_map[ticker_symbol] = np.nan

            # Dividends
            buy_date_min = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Buy Date"].min().date()
            dividends_map[ticker_symbol] = fetch_annual_dividends(ticker_symbol, buy_date_min, today_date)

            # Risk metrics
            risk_metrics_map[ticker_symbol] = compute_risk_metrics(hist_df, benchmark_returns)
        except Exception:
            names_map[ticker_symbol] = ticker_symbol
            prices_map[ticker_symbol] = np.nan
            dividends_map[ticker_symbol] = pd.Series(dtype=float)
            info_map[ticker_symbol] = {}
            risk_metrics_map[ticker_symbol] = {"volatility": np.nan, "beta": np.nan}

# Merge fetched data back into DataFrame
df_portfolio["Current"] = df_portfolio["Ticker"].map(prices_map)
df_portfolio["Value"] = df_portfolio["Current"] * df_portfolio["Shares"]
df_portfolio["Invested"] = df_portfolio["Buy Price"] * df_portfolio["Shares"]
df_portfolio["Abs Perf"] = df_portfolio["Value"] - df_portfolio["Invested"]
df_portfolio["Rel Perf"] = df_portfolio["Abs Perf"] / df_portfolio["Invested"]
df_portfolio["Name"] = df_portfolio["Ticker"].map(names_map)
df_portfolio["Sector"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("sector", "N/A"))
df_portfolio["Industry"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("industry", "N/A"))
df_portfolio["P/E"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("trailingPE", np.nan))
df_portfolio["Market Cap"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("marketCap", np.nan))
df_portfolio["Dividend Yield (%)"] = df_portfolio["Ticker"].map(
    lambda t: info_map.get(t, {}).get("dividendYield", 0.0) * 100
)
df_portfolio["P/B"] = df_portfolio["Ticker"].map(lambda t: info_map.get(t, {}).get("priceToBook", np.nan))

# Calculate portfolio totals
total_portfolio_value = df_portfolio["Value"].sum()
total_portfolio_pl = df_portfolio["Abs Perf"].sum()

# ===============================
# Portfolio Overview Section
# ===============================
if menu_option == TEXT["menu_overview"]:
    st.title(TEXT["overview_title"])

    # Display core columns
    overview_df = df_portfolio[["Name", "Ticker", "Value", "Abs Perf", "Rel Perf", "Sector", "Industry"]].copy()
    overview_df.rename(
        columns={
            "Value": TEXT["position_size_label"],
            "Abs Perf": TEXT["absolute_perf_label"],
            "Rel Perf": TEXT["relative_perf_label"],
            "Sector": "Sector",
            "Industry": "Industry",
        },
        inplace=True,
    )
    overview_df[TEXT["relative_perf_label"]] *= 100  # convert to percentage

    st.dataframe(
        overview_df.style.format(
            {
                TEXT["position_size_label"]: "€{:.2f}",
                TEXT["absolute_perf_label"]: "€{:.2f}",
                TEXT["relative_perf_label"]: "{:.2f}%",
            }
        ),
        use_container_width=True,
    )

    # Summary metrics
    st.subheader(TEXT["summary_subheader"])
    col1, col2 = st.columns(2)
    col1.metric("Total Value", f"€{total_portfolio_value:.2f}")
    col2.metric("Total P/L", f"€{total_portfolio_pl:.2f}")

    # Allocation by Value Pie Chart
    st.subheader(TEXT["allocation_by_value_label"])
    fig_value, ax_value = plt.subplots(figsize=(5, 3))
    num_positions = len(df_portfolio)
    font_size = max(6, 12 - num_positions // 2)
    wedges_value, texts_value, autotexts_value = ax_value.pie(
        df_portfolio["Value"],
        labels=df_portfolio["Ticker"],
        autopct="%1.1f%%",
        startangle=140,
        colors=PIE_CHART_COLORS,
    )
    for txt in texts_value + autotexts_value:
        txt.set_fontsize(font_size)
    ax_value.axis("equal")
    st.pyplot(fig_value)

    # Allocation by Sector Pie Chart
    st.subheader(TEXT["allocation_by_sector_label"])
    sector_alloc = (
        df_portfolio.groupby("Sector")["Value"].sum().reset_index().sort_values("Value", ascending=False)
    )
    if not sector_alloc.empty:
        fig_sector, ax_sector = plt.subplots(figsize=(5, 3))
        wedges_sector, texts_sector, autotexts_sector = ax_sector.pie(
            sector_alloc["Value"],
            labels=sector_alloc["Sector"],
            autopct="%1.1f%%",
            startangle=140,
            colors=PIE_CHART_COLORS,
        )
        for txt in texts_sector + autotexts_sector:
            txt.set_fontsize(font_size)
        ax_sector.axis("equal")
        st.pyplot(fig_sector)
    else:
        st.info("No sector data available.")

# ===============================
# Performance & Risk Analytics Section
# ===============================
elif menu_option == TEXT["menu_analytics"]:
    st.title(TEXT["analytics_title"])

    # Per-asset expanders
    returns_list = []
    for ticker_symbol in unique_tickers:
        price_history = price_histories.get(ticker_symbol, pd.DataFrame())
        daily_returns = price_history["Close"].pct_change().dropna() if not price_history.empty else pd.Series(dtype=float)
        returns_list.append(daily_returns)

        metrics = risk_metrics_map.get(ticker_symbol, {"volatility": np.nan, "beta": np.nan})
        pe_ratio = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "P/E"].iloc[0]
        market_cap_value = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Market Cap"].iloc[0]
        market_cap_str = format_market_cap(market_cap_value)
        dividend_yield = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Dividend Yield (%)"].iloc[0]
        pb_ratio = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "P/B"].iloc[0]

        with st.expander(f"{names_map.get(ticker_symbol, ticker_symbol)} ({ticker_symbol})"):
            st.write(f"P/E Ratio: {pe_ratio:.2f}")
            st.write(f"P/B Ratio: {pb_ratio:.2f}")
            st.write(f"Dividend Yield: {dividend_yield:.2f}%")
            st.write(f"Market Cap: {market_cap_str}")
            st.write(f"Volatility: {metrics['volatility']:.4f}")
            st.write(f"Beta: {metrics['beta']:.4f}")

    # Portfolio-level metrics (only if at least one returns series exists)
    if returns_list:
        # Align returns and compute portfolio returns as simple average
        portfolio_returns = pd.concat(returns_list, axis=1).mean(axis=1).dropna()

        # Sharpe & Sortino
        portfolio_mean = portfolio_returns.mean()
        portfolio_std = portfolio_returns.std()
        sharpe_ratio = (portfolio_mean / portfolio_std) * np.sqrt(252) if portfolio_std != 0 else np.nan
        downside_std = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        sortino_ratio = (portfolio_mean / downside_std) if downside_std != 0 else np.nan

        # Max Drawdown
        max_drawdown = calculate_max_drawdown(portfolio_returns)

        # CAGR & Total Return
        num_days = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days
        num_years = num_days / 365.25 if num_days > 0 else np.nan
        cumulative_return = (1 + portfolio_returns).prod()
        cagr = calculate_cagr(1, cumulative_return, num_years) if num_years > 0 else np.nan
        total_return_pct = (cumulative_return - 1) * 100 if not np.isnan(cumulative_return) else np.nan

        # Portfolio Volatility & Beta
        portfolio_volatility = portfolio_std * np.sqrt(252)
        paired_portfolio = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio_beta = linregress(paired_portfolio.iloc[:, 1], paired_portfolio.iloc[:, 0])[0] if not paired_portfolio.empty else np.nan

        # Income Yield: Last full calendar year
        previous_year = today_date.year - 1
        total_income_last_year = sum(
            dividends_map[ticker_symbol].get(previous_year, 0)
            * df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Shares"].iloc[0]
            for ticker_symbol in unique_tickers
        )
        income_yield_pct = (total_income_last_year / total_portfolio_value) * 100 if total_portfolio_value else 0.0

        # Display Summary Metrics
        st.subheader(TEXT["portfolio_summary"])
        row1_cols = st.columns(4)
        row1_cols[0].metric(TEXT["total_return_label"], f"{total_return_pct:.2f}%")
        row1_cols[1].metric(TEXT["cagr_label"], f"{cagr * 100:.2f}%" if not np.isnan(cagr) else "N/A")
        row1_cols[2].(metric(TEXT["volatility_label"], f"{portfolio_volatility:.2f"))
        row1_cols[3].metric(TEXT["beta_label"], f"{portfolio_beta:.2f")

        row2_cols = st.columns(4)
        row2_cols[0].metric(TEXT["income_yield_label"], f"{income_yield_pct:.2f}%")
        row2_cols[1].metric(TEXT["max_drawdown_label"], f"{max_drawdown:.2f}")
        row2_cols[2].metric(TEXT["sharpe_label"], f"{sharpe_ratio:.2f}")
        row2_cols[3].metric(TEXT["sortino_label"], f"{sortino_ratio:.2f}")

        # Cumulative Return Chart
        st.subheader(TEXT["cumulative_return_label"])
        benchmark_options = st.multiselect(TEXT["benchmarks_label"], ["S&P 500", "Gold (GLD)", "Bitcoin (BTC-USD)"])
        custom_tickers_input = st.text_input(TEXT["custom_tickers_label"], "", help=TEXT["custom_tickers_help"])
        custom_tickers_list = [tick.strip() for tick in custom_tickers_input.split(",") if tick.strip()]
        fig_cum, ax_cum = plt.subplots(figsize=(5, 3))
        ax_cum.plot((1 + portfolio_returns).cumprod(), label="Portfolio", linewidth=2)

        if "S&P 500" in benchmark_options:
            sp_returns = benchmark_returns
            ax_cum.plot((1 + sp_returns).cumprod(), linestyle="--", label="S&P 500")
        if "Gold (GLD)" in benchmark_options:
            gold_hist = get_price_history("GLD", analysis_start, analysis_end)
            gold_returns = gold_hist["Close"].pct_change().dropna()
            ax_cum.plot((1 + gold_returns).cumprod(), linestyle="--", label="Gold (GLD)")
        if "Bitcoin (BTC-USD)" in benchmark_options:
            btc_hist = get_price_history("BTC-USD", analysis_start, analysis_end)
            btc_returns = btc_hist["Close"].pct_change().dropna()
            ax_cum.plot((1 + btc_returns).cumprod(), linestyle="--", label="Bitcoin (BTC-USD)")

        for custom_ticker in custom_tickers_list:
            try:
                custom_hist = get_price_history(custom_ticker, analysis_start, analysis_end)
                custom_returns = custom_hist["Close"].pct_change().dropna()
                ax_cum.plot((1 + custom_returns).cumprod(), linestyle="--", label=custom_ticker)
            except Exception:
                st.warning(f"Failed to fetch data for {custom_ticker}")

        ax_cum.set_xlabel("Date")
        ax_cum.set_ylabel("Cumulative Return")
        ax_cum.legend(fontsize=8)
        st.pyplot(fig_cum)

        # Received Dividends Chart
        st.subheader(TEXT["received_dividends_label"])
        dividends_adjusted: Dict[str, pd.Series] = {}
        for ticker_symbol, dividends_series in dividends_map.items():
            shares_count = df_portfolio.loc[df_portfolio["Ticker"] == ticker_symbol, "Shares"].iloc[0]
            dividends_adjusted[ticker_symbol] = dividends_series * shares_count
        dividends_df = pd.DataFrame(dividends_adjusted).fillna(0).sort_index()
        if not dividends_df.empty:
            fig_div, ax_div = plt.subplots(figsize=(5, 3))
            dividends_df.plot(kind="bar", stacked=True, ax=ax_div, color=PIE_CHART_COLORS)
            ax_div.set_xlabel("Year")
            ax_div.set_ylabel("Dividends (€)")
            ax_div.set_title(TEXT["annual_dividends_chart_title"])
            legend = ax_div.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
            fig_div.subplots_adjust(right=0.8)
            st.pyplot(fig_div)
        else:
            st.info(TEXT["no_dividends_message"])

        # Correlation Matrix (optional)
        if st.checkbox(TEXT["show_corr_matrix_label"], value=False):
            all_returns_df = pd.DataFrame(
                {
                    ticker_symbol: price_histories[ticker_symbol]["Close"].pct_change()
                    for ticker_symbol in unique_tickers
                    if not price_histories[ticker_symbol].empty
                }
            )
            corr_matrix = all_returns_df.corr().fillna(0)
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            cax = ax_corr.matshow(corr_matrix, cmap="viridis")
            fig_corr.colorbar(cax)
            ax_corr.set_xticks(range(len(corr_matrix.columns)))
            ax_corr.set_yticks(range(len(corr_matrix.index)))
            ax_corr.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
            ax_corr.set_yticklabels(corr_matrix.index, fontsize=8)
            ax_corr.set_title("Return Correlation Matrix", pad=20)
            st.pyplot(fig_corr)
    else:
        st.info("Not enough data for performance & risk analytics.")
