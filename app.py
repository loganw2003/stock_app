# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from scipy.optimize import minimize
import math

st.set_page_config(page_title="Portfolio Analytics", layout="wide")
st.title("📊 Interactive Portfolio Analytics")

# ---------------- SIDEBAR ----------------
st.sidebar.header("User Inputs")

tickers_input = st.sidebar.text_input(
    "Enter 3–10 tickers (comma separated)",
    value="AAPL,MSFT,GOOGL"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100
rf_daily = rf_rate / 252

start_date = st.sidebar.date_input(
    "Start Date",
    value=date.today() - timedelta(days=365*5)
)

end_date = st.sidebar.date_input("End Date", value=date.today())

# ---------------- VALIDATION ----------------
if len(tickers) < 3 or len(tickers) > 10:
    st.error("Please enter between 3 and 10 tickers.")
    st.stop()

if (end_date - start_date).days < 730:
    st.error("Minimum 2-year range required.")
    st.stop()

# ---------------- DATA ----------------
@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    data = yf.download(tickers + ["^GSPC"], start=start, end=end)["Adj Close"]
    return data

prices = load_data(tickers, start_date, end_date)

returns = prices.pct_change().dropna()

# ---------------- FUNCTIONS ----------------
def annualized_return(r):
    return r.mean() * 252

def annualized_vol(r):
    return r.std() * np.sqrt(252)

def sharpe_ratio(r):
    return (annualized_return(r) - rf_rate) / annualized_vol(r)

def sortino_ratio(r):
    downside = r[r < rf_daily]
    return (annualized_return(r) - rf_rate) / (downside.std() * np.sqrt(252))

def portfolio_stats(w, mean, cov):
    ret = np.dot(w, mean)
    vol = np.sqrt(w.T @ cov @ w)
    sharpe = (ret - rf_rate) / vol
    return ret, vol, sharpe

def max_drawdown(series):
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Returns",
    "⚠️ Risk",
    "🔗 Correlation",
    "📊 Portfolio",
    "🔍 Sensitivity"
])

# =======================
# RETURNS
# =======================
with tab1:
    st.subheader("Summary Statistics")

    stats = pd.DataFrame({
        "Return": returns.apply(annualized_return),
        "Volatility": returns.apply(annualized_vol),
        "Skew": returns.skew(),
        "Kurtosis": returns.kurtosis(),
        "Min": returns.min(),
        "Max": returns.max()
    })

    st.dataframe(stats)

    st.subheader("Cumulative Wealth")
    wealth = (1 + returns).cumprod() * 10000
    st.line_chart(wealth)

# =======================
# RISK
# =======================
with tab2:
    window = st.slider("Rolling Vol Window", 30, 120, 60)

    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    st.line_chart(rolling_vol)

    stock = st.selectbox("Drawdown Stock", returns.columns)
    dd = (1 + returns[stock]).cumprod()
    dd = (dd - dd.cummax()) / dd.cummax()

    st.line_chart(dd)
    st.metric("Max Drawdown", f"{dd.min():.2%}")

    st.subheader("Sharpe & Sortino")
    ratios = pd.DataFrame({
        "Sharpe": returns.apply(sharpe_ratio),
        "Sortino": returns.apply(sortino_ratio)
    })
    st.dataframe(ratios)

# =======================
# CORRELATION
# =======================
with tab3:
    st.dataframe(returns.corr())

    s1 = st.selectbox("Stock 1", returns.columns)
    s2 = st.selectbox("Stock 2", returns.columns)

    rolling_corr = returns[s1].rolling(60).corr(returns[s2])
    st.line_chart(rolling_corr)

# =======================
# PORTFOLIO
# =======================
with tab4:
    mean = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(mean)

    bounds = tuple((0,1) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w)-1}
    init = np.ones(n)/n

    # GMV
    def gmv(w):
        return np.sqrt(w.T @ cov @ w)

    gmv_res = minimize(gmv, init, bounds=bounds, constraints=constraints)
    w_gmv = gmv_res.x

    # Tangency
    def neg_sharpe(w):
        return -portfolio_stats(w, mean, cov)[2]

    tan_res = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)
    w_tan = tan_res.x

    # Custom portfolio
    st.subheader("Custom Portfolio")
    sliders = np.array([
        st.slider(f"{t}", 0.0, 1.0, 1.0/n)
        for t in tickers
    ])
    w_custom = sliders / sliders.sum()

    st.write("Normalized Weights:", pd.Series(w_custom, index=tickers))

    # Risk contribution
    def risk_contribution(w):
        port_var = w.T @ cov @ w
        return (w * (cov @ w)) / port_var

    prc = risk_contribution(w_tan)

    st.subheader("Risk Contribution (Tangency)")
    st.bar_chart(pd.Series(prc, index=tickers))

    # Efficient Frontier
    target_returns = np.linspace(mean.min(), mean.max(), 50)
    vols = []

    for tr in target_returns:
        cons = [
            {'type':'eq','fun':lambda w: np.sum(w)-1},
            {'type':'eq','fun':lambda w: np.dot(w, mean)-tr}
        ]
        res = minimize(gmv, init, bounds=bounds, constraints=cons)
        vols.append(res.fun)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=target_returns, name="Efficient Frontier"))
    st.plotly_chart(fig)

# =======================
# SENSITIVITY
# =======================
with tab5:
    st.subheader("Estimation Window Sensitivity")

    options = ["1Y", "3Y", "5Y", "Full"]
    results = []

    for opt in options:
        if opt == "Full":
            sub = returns
        else:
            days = int(opt[0]) * 252
            if len(returns) < days:
                continue
            sub = returns.tail(days)

        mean_s = sub.mean() * 252
        cov_s = sub.cov() * 252

        res = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)
        w = res.x
        r, v, s = portfolio_stats(w, mean_s, cov_s)

        results.append([opt, r, v, s])

    df_sens = pd.DataFrame(results, columns=["Window","Return","Vol","Sharpe"])
    st.dataframe(df_sens)

# ---------------- ABOUT ----------------
with st.expander("About"):
    st.write("""
    - Simple returns used
    - Annualization uses 252 trading days
    - Risk-free rate converted to daily
    - No short selling
    - Data: Yahoo Finance
    """)