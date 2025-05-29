# Import the necessary libraries (like toolkits for specific tasks)
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from io import StringIO

# Helper functions

def calculate_cagr(start_value: float, end_value: float, periods: float) -> float:
    return (end_value / start_value) ** (1 / periods) - 1

def calculate_max_drawdown(series: pd.Series) -> float:
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def fetch_annual_dividends(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
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

# App setup
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📊 Portfolio Menu")
    menu = st.radio("Navigation", ["📁 Upload CSV", "📈 Portfolio Overview", "📉 Performance & Risk Analytics"])
    if "last_updated" in st.session_state:
        st.caption(f"Last updated: {st.session_state['last_updated']}")
    if "portfolio_filename" in st.session_state:
        st.caption(f"File: {st.session_state['portfolio_filename']}")

# Session state init
if "portfolio_file" not in st.session_state:
    st.session_state.portfolio_file = None
    st.session_state.portfolio_filename = None

# Upload
if menu == "📁 Upload CSV":
    st.title("Upload Portfolio CSV")
    st.info("Columns: Ticker, Shares, Buy Price, Buy Date")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        st.session_state.portfolio_file = uploaded.read()
        st.session_state.portfolio_filename = uploaded.name
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("File uploaded successfully.")

# Process file
file_content = st.session_state.get('portfolio_file')
if file_content and menu != "📁 Upload CSV":
    df = pd.read_csv(StringIO(file_content.decode('utf-8')))
    df['Buy Date'] = pd.to_datetime(df['Buy Date'])
    today = datetime.today()
    required = ["Ticker","Shares","Buy Price","Buy Date"]
    if not all(c in df.columns for c in required):
        st.error("CSV must contain: Ticker, Shares, Buy Price, Buy Date")
        st.stop()

    # Fetch data
    names, prices, dividends_map, history_map = {}, [], {}, {}
    for _, row in df.iterrows():
        t, bd = row['Ticker'], row['Buy Date']
        tk = yf.Ticker(t)
        history = tk.history(start=bd, end=today)
        history_map[t] = history
        info = tk.info
        names[t] = info.get('shortName', t)
        prices.append(history['Close'].iloc[-1] if not history.empty else np.nan)
        dividends_map[t] = fetch_annual_dividends(t, bd, today)
        df.loc[df['Ticker']==t, 'P/E'] = info.get('trailingPE', np.nan)
        df.loc[df['Ticker']==t, 'Market Cap'] = info.get('marketCap', np.nan)

    # Calculate metrics
    df['Current'] = prices
    df['Value'] = df['Current']*df['Shares']
    df['Invested'] = df['Shares']*df['Buy Price']
    df['Abs Perf'] = df['Value']-df['Invested']
    df['Rel Perf'] = df['Abs Perf']/df['Invested']
    df['Name'] = df['Ticker'].map(names)
    total_value = df['Value'].sum()
    total_pl = df['Abs Perf'].sum()

    # Overview
    if menu == "📈 Portfolio Overview":
        st.title("Portfolio Overview")
        cols = ['Name','Ticker','Value','Abs Perf','Rel Perf','P/E','Market Cap']
        display = df[cols].copy()
        display.rename(columns={
            'Value':'Position Size (€)',
            'Abs Perf':'Absolute Perf (€)',
            'Rel Perf':'Relative Perf (%)',
            'Market Cap':'Market Cap (€)'
        }, inplace=True)
        display['Relative Perf (%)']*=100
        display['Market Cap (€)']=display['Market Cap (€)'].apply(lambda x:f"€{x:,.0f}")
        st.dataframe(display.style.format({
            'Position Size (€)':'€{:.2f}',
            'Absolute Perf (€)':'€{:.2f}',
            'Relative Perf (%)':'{:.2f}%',
            'P/E':'{:.2f}'
        }), use_container_width=True)

        st.subheader("Summary")
        c1,c2 = st.columns(2)
        c1.metric("Total Value", f"€{total_value:.2f}")
        c2.metric("Total P/L", f"€{total_pl:.2f}")

        st.subheader("Allocation by Value")
        fig,ax=plt.subplots(figsize=(5,3))
        colors=plt.get_cmap('tab20').colors
        ax.pie(df['Value'],labels=df['Ticker'],autopct='%1.1f%%',startangle=140,colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

    # Performance & Risk
    elif menu == "📉 Performance & Risk Analytics":
        st.title("Performance & Risk Analytics")

        # Per-asset metrics
        returns_list=[]
        start_date=df['Buy Date'].min().date()
        bench= yf.Ticker('^GSPC').history(start=start_date,end=today.date())['Close'].pct_change()
        for t,h in history_map.items():
            r=h['Close'].pct_change()
            returns_list.append(r)
            vol=r.std()*np.sqrt(252)
            mdd=calculate_max_drawdown(r)
            pr=pd.concat([r,bench],axis=1).dropna()
            beta=linregress(pr.iloc[:,1],pr.iloc[:,0])[0] if not pr.empty else np.nan
            pe=df.loc[df['Ticker']==t,'P/E'].iloc[0]
            mcap=df.loc[df['Ticker']==t,'Market Cap'].iloc[0]
            with st.expander(f"{names[t]} ({t})"):
                st.write(f"P/E: {pe:.2f}")
                st.write(f"Market Cap: €{mcap:,.0f}")
                st.write(f"Volatility: {vol:.4f}")
                st.write(f"Max Drawdown: {mdd:.4f}")
                st.write(f"Beta: {beta:.4f}")

        # Portfolio metrics
        if returns_list:
            prt=pd.concat(returns_list,axis=1).mean(axis=1)
            sharpe=prt.mean()/prt.std()*np.sqrt(252)
            downside=prt[prt<0].std()*np.sqrt(252)
            sortino=prt.mean()/downside if downside else np.nan
            mdd_p=calculate_max_drawdown(prt)
            days=(prt.index[-1]-prt.index[0]).days
            yrs=days/365.25
            cum_ret=(1+prt).prod()
            cagr=calculate_cagr(1,cum_ret,yrs)
            tot_ret=(cum_ret-1)*100
            vol_p=prt.std()*np.sqrt(252)
            bp=pd.concat([prt,bench],axis=1).dropna()
            beta_p=linregress(bp.iloc[:,1],bp.iloc[:,0])[0] if not bp.empty else np.nan

            st.subheader("Portfolio Summary")
            r1c1,r1c2,r1c3,r1c4=st.columns(4)
            r1c1.metric("Total Return (%)",f"{tot_ret:.2f}%")
            r1c2.metric("CAGR (%)",f"{cagr*100:.2f}%")
            r1c3.metric("Volatility",f"{vol_p:.2f}")
            r1c4.metric("Beta",f"{beta_p:.2f}")
            r2c1,r2c2,r2c3=st.columns(3)
            r2c1.metric("Max Drawdown",f"{mdd_p:.2f}")
            r2c2.metric("Sharpe",round(sharpe,3))
            r2c3.metric("Sortino",round(sortino,3))

            st.subheader("Cumulative Return")
            cb=st.multiselect("Benchmarks",["S&P 500","Gold (GLD)","Bitcoin (BTC-USD)"])
            custom=st.text_input("Custom tickers, comma-separated","")
            cl=[x.strip() for x in custom.split(',') if x.strip()]
            fig3,ax3=plt.subplots(figsize=(5,3))
            ax3.plot((1+prt).cumprod(),label="Portfolio",linewidth=2)
            if "S&P 500" in cb:
                sp=(1+bench).cumprod()
                ax3.plot(sp,linestyle='--',label="S&P 500")
            if "Gold (GLD)" in cb:
                g=yf.Ticker("GLD").history(start=start_date,end=today)['Close'].pct_change()
                ax3.plot((1+g).cumprod(),linestyle='--',label="Gold (GLD)")
            if "Bitcoin (BTC-USD)" in cb:
                b=yf.Ticker("BTC-USD").history(start=start_date,end=today)['Close'].pct_change()
                ax3.plot((1+b).cumprod(),linestyle='--',label="Bitcoin")
            for x in cl:
                try:
                    temp=yf.Ticker(x).history(start=start_date,end=today)['Close'].pct_change()
                    ax3.plot((1+temp).cumprod(),linestyle='--',label=x)
                except:
                    st.warning(f"{x} fetch failed")
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Cumulative Return')
            ax3.legend(fontsize=8)
            st.pyplot(fig3)

            # Dividends in P&R
            st.subheader("Received Dividends")
            adj={}
            for t,s in dividends_map.items():
                adj[t]=s*df.loc[df['Ticker']==t,'Shares'].iloc[0]
            ddf=pd.DataFrame(adj).fillna(0).sort_index()
            if not ddf.empty:
                fig4,ax4=plt.subplots(figsize=(5,3))
                ddf.plot(kind='bar',stacked=True,ax=ax4,color=plt.get_cmap('tab20').colors)
                ax4.set_xlabel('Year')
                ax4.set_ylabel('Dividends (€)')
                ax4.set_title('Annual Dividends Received')
                lg=ax4.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
                fig4.subplots_adjust(right=0.8)
                st.pyplot(fig4)
            else:
                st.info("No dividends.")
