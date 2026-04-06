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
    st.warning("Please enter between 3 and 10 tickers.")
    st.stop()

if (end_date - start_date).days < 730:
    st.warning("Minimum 2-year date range required.")
    st.stop()

# ---------------- DATA ----------------
@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    try:
        data = yf.download(tickers + ["^GSPC"], start=start, end=end, progress=False)

        if data.empty:
            return None, ["No data returned"]

        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data = data["Adj Close"]

        return data, []

    except Exception as e:
        return None, [str(e)]

with st.spinner("Downloading market data..."):
    prices, errors = load_data(tickers, start_date, end_date)

if errors:
    st.error(f"Download error: {errors}")
    st.stop()

if prices is None or prices.empty:
    st.error("No valid data returned. Try different tickers.")
    st.stop()

# ---------------- CLEAN DATA ----------------
# Drop columns with >5% missing
threshold = int(0.95 * len(prices))
prices = prices.dropna(axis=1, thresh=threshold)

if len(prices.columns) < 3:
    st.error("Too many tickers had missing data. Need at least 3 valid stocks.")
    st.stop()

returns = prices.pct_change().dropna()

if returns.empty:
    st.error("Returns calculation failed due to missing data.")
    st.stop()

# ---------------- FUNCTIONS ----------------
def ann_return(r): return r.mean() * 252
def ann_vol(r): return r.std() * np.sqrt(252)

def sharpe(r):
    return (ann_return(r) - rf_rate) / ann_vol(r)

def sortino(r):
    downside = r[r < rf_daily]
    if len(downside) == 0:
        return np.nan
    return (ann_return(r) - rf_rate) / (downside.std() * np.sqrt(252))

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

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
        "Return": returns.apply(ann_return),
        "Volatility": returns.apply(ann_vol),
        "Sharpe": returns.apply(sharpe),
        "Sortino": returns.apply(sortino)
    })

    st.dataframe(stats)

    st.subheader("Cumulative Wealth ($10,000)")
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

# =======================
# CORRELATION
# =======================
with tab3:
    st.subheader("Correlation Matrix")
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

    bounds = [(0,1)] * n
    constraints = {'type':'eq','fun':lambda w: np.sum(w)-1}
    init = np.ones(n)/n

    # GMV
    def gmv(w): return np.sqrt(w.T @ cov @ w)

    gmv_res = minimize(gmv, init, bounds=bounds, constraints=constraints)

    if not gmv_res.success:
        st.error("GMV optimization failed")
        st.stop()

    w_gmv = gmv_res.x

    # Tangency
    def neg_sharpe(w):
        ret = np.dot(w, mean)
        vol = np.sqrt(w.T @ cov @ w)
        return -(ret - rf_rate)/vol

    tan_res = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)

    if not tan_res.success:
        st.error("Tangency optimization failed")
        st.stop()

    w_tan = tan_res.x

    st.subheader("Tangency Weights")
    st.write(pd.Series(w_tan, index=returns.columns))

    # Risk contribution
    port_var = w_tan.T @ cov @ w_tan
    prc = (w_tan * (cov @ w_tan)) / port_var

    st.subheader("Risk Contribution")
    st.bar_chart(pd.Series(prc, index=returns.columns))

    # Custom portfolio
    st.subheader("Custom Portfolio")

    sliders = np.array([
        st.slider(f"{t}", 0.0, 1.0, 1.0/n)
        for t in returns.columns
    ])

    w_custom = sliders / sliders.sum()
    st.write("Normalized:", pd.Series(w_custom, index=returns.columns))

# =======================
# SENSITIVITY
# =======================
with tab5:
    st.subheader("Estimation Window Sensitivity")

    options = ["1Y","3Y","5Y","Full"]
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

        res = minimize(
            lambda w: -(np.dot(w, mean_s) - rf_rate)/np.sqrt(w.T@cov_s@w),
            init,
            bounds=bounds,
            constraints=constraints
        )

        if res.success:
            w = res.x
            r = np.dot(w, mean_s)
            v = np.sqrt(w.T @ cov_s @ w)
            s = (r - rf_rate)/v
            results.append([opt, r, v, s])

    st.dataframe(pd.DataFrame(results, columns=["Window","Return","Vol","Sharpe"]))

# ---------------- ABOUT ----------------
with st.expander("About"):
    st.write("""
    - Uses adjusted close prices from Yahoo Finance
    - Returns are simple (not log)
    - Annualization uses 252 trading days
    - No short selling allowed
    - Optimization via scipy.optimize
    """)