# Portfolio Analyzer (Streamlit App)

This Streamlit web app allows users to upload a portfolio in CSV format, analyze individual assets, and evaluate overall performance and risk metrics.

## Features

- Upload a CSV file containing your portfolio
- View portfolio allocation and performance
- Per-Asset metrics: Volatility, Max Drawdown, Beta vs. S&P500
- Portfolio metrics: Sharpe Ratio, Sortino Ratio, CAGR
- Visual charts and cumulative return tracking

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## CSV Format

Your uploaded CSV should contain the following columns:

- `Ticker`
- `Shares`
- `Buy Price`
- `Buy Date` (format: YYYY-MM-DD)

## Run the App

```bash
python -m streamlit run app.py
```

