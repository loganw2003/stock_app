import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from scipy.optimize import minimize
from scipy import stats

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(page_title="Interactive Portfolio Analytics", layout="wide")
st.title("📊 Interactive Portfolio Analytics Application")

TRADING_DAYS = 252
BENCHMARK = "^GSPC"

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def format_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "N/A"


def annualized_return(series: pd.Series) -> float:
    return series.mean() * TRADING_DAYS


def annualized_vol(series: pd.Series) -> float:
    return series.std() * np.sqrt(TRADING_DAYS)


def downside_deviation(series: pd.Series, rf_annual: float) -> float:
    rf_daily = rf_annual / TRADING_DAYS
    downside = series[series < rf_daily] - rf_daily
    if len(downside) == 0:
        return np.nan
    return np.sqrt((downside.pow(2).mean()) * TRADING_DAYS)


def sharpe_ratio(series: pd.Series, rf_annual: float) -> float:
    vol = annualized_vol(series)
    if vol == 0 or pd.isna(vol):
        return np.nan
    return (annualized_return(series) - rf_annual) / vol


def sortino_ratio(series: pd.Series, rf_annual: float) -> float:
    dd = downside_deviation(series, rf_annual)
    if dd == 0 or pd.isna(dd):
        return np.nan
    return (annualized_return(series) - rf_annual) / dd


def max_drawdown_from_returns(series: pd.Series) -> float:
    wealth = (1 + series).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1
    return drawdown.min()


def summary_stats(returns_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Annualized Mean Return": returns_df.mean() * TRADING_DAYS,
        "Annualized Volatility": returns_df.std() * np.sqrt(TRADING_DAYS),
        "Skewness": returns_df.skew(),
        "Kurtosis": returns_df.kurtosis(),
        "Min Daily Return": returns_df.min(),
        "Max Daily Return": returns_df.max(),
    })


def risk_adjusted_table(returns_df: pd.DataFrame, rf_annual: float) -> pd.DataFrame:
    return pd.DataFrame({
        "Sharpe Ratio": returns_df.apply(lambda s: sharpe_ratio(s, rf_annual)),
        "Sortino Ratio": returns_df.apply(lambda s: sortino_ratio(s, rf_annual)),
    })


def portfolio_return(weights: np.ndarray, mean_returns_annual: np.ndarray) -> float:
    return float(weights @ mean_returns_annual)


def portfolio_vol(weights: np.ndarray, cov_annual: np.ndarray) -> float:
    return float(np.sqrt(weights.T @ cov_annual @ weights))


def portfolio_sharpe(weights: np.ndarray, mean_returns_annual: np.ndarray, cov_annual: np.ndarray, rf_annual: float) -> float:
    vol = portfolio_vol(weights, cov_annual)
    if vol == 0:
        return np.nan
    return (portfolio_return(weights, mean_returns_annual) - rf_annual) / vol


def risk_contribution(weights: np.ndarray, cov_annual: np.ndarray) -> np.ndarray:
    port_var = float(weights.T @ cov_annual @ weights)
    if port_var == 0:
        return np.full_like(weights, np.nan)
    return (weights * (cov_annual @ weights)) / port_var


def optimize_gmv(cov_annual: np.ndarray):
    n = cov_annual.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        lambda w: portfolio_vol(w, cov_annual),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def optimize_tangency(mean_returns_annual: np.ndarray, cov_annual: np.ndarray, rf_annual: float):
    n = len(mean_returns_annual)
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def negative_sharpe(w):
        vol = portfolio_vol(w, cov_annual)
        if vol == 0:
            return 1e9
        ret = portfolio_return(w, mean_returns_annual)
        return -((ret - rf_annual) / vol)

    result = minimize(
        negative_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def optimize_target_return(target_return: float, mean_returns_annual: np.ndarray, cov_annual: np.ndarray):
    n = len(mean_returns_annual)
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: (w @ mean_returns_annual) - target_return},
    ]

    result = minimize(
        lambda w: portfolio_vol(w, cov_annual),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def efficient_frontier(mean_returns_annual: np.ndarray, cov_annual: np.ndarray, n_points: int = 25) -> pd.DataFrame:
    ret_min = float(np.min(mean_returns_annual))
    ret_max = float(np.max(mean_returns_annual))
    targets = np.linspace(ret_min, ret_max, n_points)

    rows = []
    for target in targets:
        res = optimize_target_return(target, mean_returns_annual, cov_annual)
        if res.success:
            rows.append({
                "Target Return": target,
                "Volatility": portfolio_vol(res.x, cov_annual),
                "Weights": res.x,
            })
    return pd.DataFrame(rows)


def build_portfolio_metrics(weights: np.ndarray, asset_returns: pd.DataFrame, rf_annual: float) -> dict:
    mean_returns_annual = (asset_returns.mean() * TRADING_DAYS).values
    cov_annual = (asset_returns.cov() * TRADING_DAYS).values
    port_daily = asset_returns @ weights

    return {
        "Annualized Return": portfolio_return(weights, mean_returns_annual),
        "Annualized Volatility": portfolio_vol(weights, cov_annual),
        "Sharpe Ratio": sharpe_ratio(port_daily, rf_annual),
        "Sortino Ratio": sortino_ratio(port_daily, rf_annual),
        "Max Drawdown": max_drawdown_from_returns(port_daily),
        "Daily Returns": port_daily,
    }


def make_line_chart(df: pd.DataFrame, title: str, y_title: str):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        template="plotly_white",
        height=500,
    )
    return fig


def make_bar_chart(series: pd.Series, title: str, y_title: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=series.index.tolist(), y=series.values))
    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title=y_title,
        template="plotly_white",
        height=450,
    )
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def compute_sensitivity(asset_returns: pd.DataFrame, asset_cols: list[str], rf_annual: float):
    sens_rows = []
    weight_rows = []

    total_obs = len(asset_returns)
    available_options = []

    if total_obs >= TRADING_DAYS * 1:
        available_options.append("1 Year")
    if total_obs >= TRADING_DAYS * 3:
        available_options.append("3 Years")
    if total_obs >= TRADING_DAYS * 5:
        available_options.append("5 Years")
    available_options.append("Full Sample")

    for option in available_options:
        if option == "1 Year":
            sub = asset_returns.tail(TRADING_DAYS * 1)
        elif option == "3 Years":
            sub = asset_returns.tail(TRADING_DAYS * 3)
        elif option == "5 Years":
            sub = asset_returns.tail(TRADING_DAYS * 5)
        else:
            sub = asset_returns.copy()

        sub_mean = sub.mean().values * TRADING_DAYS
        sub_cov = sub.cov().values * TRADING_DAYS

        sub_gmv = optimize_gmv(sub_cov)
        sub_tan = optimize_tangency(sub_mean, sub_cov, rf_annual)

        if sub_gmv.success:
            gmv_w = sub_gmv.x
            sens_rows.append({
                "Window": option,
                "Portfolio": "GMV",
                "Annualized Return": portfolio_return(gmv_w, sub_mean),
                "Annualized Volatility": portfolio_vol(gmv_w, sub_cov),
                "Sharpe Ratio": np.nan,
                "Weights": ", ".join([f"{asset_cols[i]}={gmv_w[i]:.2%}" for i in range(len(asset_cols))]),
            })
            for i, ticker in enumerate(asset_cols):
                weight_rows.append({
                    "Window": option,
                    "Portfolio": "GMV",
                    "Ticker": ticker,
                    "Weight": gmv_w[i],
                })

        if sub_tan.success:
            tan_w = sub_tan.x
            sens_rows.append({
                "Window": option,
                "Portfolio": "Tangency",
                "Annualized Return": portfolio_return(tan_w, sub_mean),
                "Annualized Volatility": portfolio_vol(tan_w, sub_cov),
                "Sharpe Ratio": portfolio_sharpe(tan_w, sub_mean, sub_cov, rf_annual),
                "Weights": ", ".join([f"{asset_cols[i]}={tan_w[i]:.2%}" for i in range(len(asset_cols))]),
            })
            for i, ticker in enumerate(asset_cols):
                weight_rows.append({
                    "Window": option,
                    "Portfolio": "Tangency",
                    "Ticker": ticker,
                    "Weight": tan_w[i],
                })

    sensitivity_df = pd.DataFrame(sens_rows)
    weights_long = pd.DataFrame(weight_rows)
    return sensitivity_df, weights_long


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.header("Inputs")

default_start = date.today() - timedelta(days=365 * 5)
default_end = date.today() - timedelta(days=1)

tickers_input = st.sidebar.text_input(
    "Enter 3–10 stock tickers (comma separated)",
    value="AAPL,MSFT,GOOGL,NVDA"
)

start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=default_end, min_value=date(1970, 1, 1))

rf_annual = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1
) / 100

rolling_vol_window = st.sidebar.selectbox("Rolling Volatility Window", [30, 60, 90, 120], index=1)
rolling_corr_window = st.sidebar.selectbox("Rolling Correlation Window", [30, 60, 90, 120], index=1)

user_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
user_tickers = list(dict.fromkeys(user_tickers))

# -------------------------------------------------------
# Validation
# -------------------------------------------------------
if len(user_tickers) < 3 or len(user_tickers) > 10:
    st.error("Please enter between 3 and 10 ticker symbols.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < 730:
    st.error("Please select at least a 2-year date range.")
    st.stop()

# -------------------------------------------------------
# Data download
# -------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start, end):
    try:
        fetch_end = pd.to_datetime(end) + pd.Timedelta(days=1)

        raw_assets = yf.download(
            tickers,
            start=start,
            end=fetch_end,
            auto_adjust=False,
            group_by="column",
            progress=False,
            threads=False,
        )

        if raw_assets is None or raw_assets.empty:
            return None, ["No data returned for the selected stock tickers."], [], []

        if not isinstance(raw_assets.columns, pd.MultiIndex):
            return None, [f"Unexpected stock data format: {list(raw_assets.columns)}"], [], []

        asset_level0 = raw_assets.columns.get_level_values(0)
        if "Adj Close" in asset_level0:
            asset_prices = raw_assets["Adj Close"].copy()
        elif "Close" in asset_level0:
            asset_prices = raw_assets["Close"].copy()
        else:
            return None, [f"Expected 'Adj Close' or 'Close' for stocks. Found: {sorted(set(asset_level0))}"], [], []

        if isinstance(asset_prices, pd.Series):
            asset_prices = asset_prices.to_frame()

        expected_assets = tickers.copy()
        existing_assets = [c for c in expected_assets if c in asset_prices.columns]
        missing_assets = [c for c in expected_assets if c not in asset_prices.columns]

        asset_prices = asset_prices[existing_assets].sort_index()

        all_empty_assets = asset_prices.columns[asset_prices.isna().all()].tolist()
        asset_prices = asset_prices.drop(columns=all_empty_assets, errors="ignore")

        # First fallback: retry still-missing tickers one by one with download()
        retried_frames = []
        for tkr in missing_assets:
            try:
                single = yf.download(
                    tkr,
                    start=start,
                    end=fetch_end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )

                if single is not None and not single.empty:
                    if isinstance(single.columns, pd.MultiIndex):
                        lvl0 = single.columns.get_level_values(0)
                        if "Adj Close" in lvl0:
                            s = single["Adj Close"]
                        elif "Close" in lvl0:
                            s = single["Close"]
                        else:
                            continue
                    else:
                        if "Adj Close" in single.columns:
                            s = single["Adj Close"]
                        elif "Close" in single.columns:
                            s = single["Close"]
                        else:
                            continue

                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]

                    retried_frames.append(s.rename(tkr))
            except Exception:
                pass

        if retried_frames:
            retry_df = pd.concat(retried_frames, axis=1)
            asset_prices = pd.concat([asset_prices, retry_df], axis=1)
            asset_prices = asset_prices.loc[:, ~asset_prices.columns.duplicated()]
            asset_prices = asset_prices.sort_index()

        # Second fallback: Ticker.history() for any still-missing symbols
        still_missing = [c for c in expected_assets if c not in asset_prices.columns]

        history_retry_frames = []
        for tkr in still_missing:
            try:
                hist = yf.Ticker(tkr).history(
                    start=start,
                    end=fetch_end,
                    auto_adjust=False
                )

                if hist is not None and not hist.empty:
                    if "Adj Close" in hist.columns:
                        s = hist["Adj Close"]
                    elif "Close" in hist.columns:
                        s = hist["Close"]
                    else:
                        continue

                    history_retry_frames.append(s.rename(tkr))
            except Exception:
                pass

        if history_retry_frames:
            history_retry_df = pd.concat(history_retry_frames, axis=1)
            asset_prices = pd.concat([asset_prices, history_retry_df], axis=1)
            asset_prices = asset_prices.loc[:, ~asset_prices.columns.duplicated()]
            asset_prices = asset_prices.sort_index()

        all_empty_assets = asset_prices.columns[asset_prices.isna().all()].tolist()
        asset_prices = asset_prices.drop(columns=all_empty_assets, errors="ignore")

        invalid_tickers = [c for c in expected_assets if c not in asset_prices.columns]
        invalid_tickers = list(dict.fromkeys(invalid_tickers + all_empty_assets))

        if not asset_prices.empty:
            missing_fraction = asset_prices.isna().mean()
            dropped_for_missing = missing_fraction[missing_fraction > 0.05].index.tolist()
            keep_assets = [c for c in asset_prices.columns if c not in dropped_for_missing]
            asset_prices = asset_prices[keep_assets]
        else:
            dropped_for_missing = []

        if asset_prices.shape[1] < 3:
            return None, ["Fewer than 3 valid stock tickers remain after cleaning."], dropped_for_missing, invalid_tickers

        raw_benchmark = yf.download(
            BENCHMARK,
            start=start,
            end=fetch_end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if raw_benchmark is None or raw_benchmark.empty:
            return None, [f"Benchmark ({BENCHMARK}) failed to download from Yahoo Finance."], dropped_for_missing, invalid_tickers

        if isinstance(raw_benchmark.columns, pd.MultiIndex):
            bench_level0 = raw_benchmark.columns.get_level_values(0)
            if "Adj Close" in bench_level0:
                benchmark_prices = raw_benchmark["Adj Close"].copy()
            elif "Close" in bench_level0:
                benchmark_prices = raw_benchmark["Close"].copy()
            else:
                return None, [f"Expected 'Adj Close' or 'Close' for benchmark. Found: {sorted(set(bench_level0))}"], dropped_for_missing, invalid_tickers
        else:
            if "Adj Close" in raw_benchmark.columns:
                benchmark_prices = raw_benchmark["Adj Close"].copy()
            elif "Close" in raw_benchmark.columns:
                benchmark_prices = raw_benchmark["Close"].copy()
            else:
                return None, [f"Expected 'Adj Close' or 'Close' for benchmark. Found: {list(raw_benchmark.columns)}"], dropped_for_missing, invalid_tickers

        if isinstance(benchmark_prices, pd.DataFrame):
            benchmark_prices = benchmark_prices.iloc[:, 0]

        benchmark_prices = benchmark_prices.rename(BENCHMARK).to_frame()

        prices = pd.concat([asset_prices, benchmark_prices], axis=1)
        prices = prices.dropna(how="any")

        if prices.empty:
            return None, ["After aligning stock and benchmark dates, no overlapping observations remained."], dropped_for_missing, invalid_tickers

        return prices, [], dropped_for_missing, invalid_tickers

    except Exception as e:
        return None, [str(e)], [], []


with st.spinner("Downloading and analyzing data..."):
    prices, download_errors, dropped_for_missing, invalid_tickers = load_data(user_tickers, start_date, end_date)

if download_errors:
    st.error("Download error: " + " | ".join(download_errors))
    st.stop()

if invalid_tickers:
    st.warning("These ticker(s) failed to download or returned no usable data: " + ", ".join(invalid_tickers))

if dropped_for_missing:
    st.warning(
        "These ticker(s) were dropped because they had more than 5% missing values over the selected period: "
        + ", ".join(dropped_for_missing)
    )

if prices is None or prices.empty:
    st.error("No usable price data was returned. Try different tickers or a different date range.")
    st.stop()

asset_cols = [c for c in prices.columns if c != BENCHMARK]
if len(asset_cols) < 3:
    st.error("Fewer than 3 valid stock tickers remain after cleaning. Please try different tickers.")
    st.stop()

returns = prices.pct_change().dropna()
asset_returns = returns[asset_cols]
benchmark_returns = returns[BENCHMARK]
all_returns = returns[asset_cols + [BENCHMARK]]

if asset_returns.empty:
    st.error("Return calculation failed after data cleaning.")
    st.stop()

# -------------------------------------------------------
# Portfolio calculations
# -------------------------------------------------------
mean_returns_annual = asset_returns.mean().values * TRADING_DAYS
cov_annual = asset_returns.cov().values * TRADING_DAYS
n_assets = len(asset_cols)

equal_weights = np.ones(n_assets) / n_assets
eq_metrics = build_portfolio_metrics(equal_weights, asset_returns, rf_annual)

gmv_result = optimize_gmv(cov_annual)
tan_result = optimize_tangency(mean_returns_annual, cov_annual, rf_annual)

if not gmv_result.success:
    st.error("Global Minimum Variance optimization failed.")
    st.stop()

if not tan_result.success:
    st.error("Tangency portfolio optimization failed.")
    st.stop()

gmv_weights = gmv_result.x
tan_weights = tan_result.x

gmv_metrics = build_portfolio_metrics(gmv_weights, asset_returns, rf_annual)
tan_metrics = build_portfolio_metrics(tan_weights, asset_returns, rf_annual)

gmv_prc = pd.Series(risk_contribution(gmv_weights, cov_annual), index=asset_cols)
tan_prc = pd.Series(risk_contribution(tan_weights, cov_annual), index=asset_cols)

frontier_df = efficient_frontier(mean_returns_annual, cov_annual, n_points=25)
sensitivity_df, weights_long = compute_sensitivity(asset_returns, asset_cols, rf_annual)

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Inputs & Data",
    "Returns & Exploratory Analysis",
    "Risk Analysis",
    "Correlation & Covariance",
    "Portfolio Construction & Optimization",
    "Estimation Window Sensitivity",
])

with tab1:
    st.subheader("Data Summary")

    info_df = pd.DataFrame({
        "Metric": [
            "Valid Asset Tickers",
            "Benchmark",
            "Start Date",
            "End Date",
            "Observations",
            "Risk-Free Rate",
        ],
        "Value": [
            ", ".join(asset_cols),
            BENCHMARK,
            str(prices.index.min().date()),
            str(prices.index.max().date()),
            len(prices),
            f"{rf_annual:.2%}",
        ],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    with st.expander("Debug download info"):
        st.write("Requested stock tickers:", user_tickers)
        st.write("Benchmark:", BENCHMARK)
        st.write("Start date:", start_date)
        st.write("End date:", end_date)
        st.write("Prices shape:", prices.shape)
        st.write("Final asset columns after cleaning:", asset_cols)
        st.write("Invalid stock tickers:", invalid_tickers)
        st.write("Dropped for missing:", dropped_for_missing)
        st.write("Download errors:", download_errors)

    st.subheader("Adjusted Close Prices")
    st.plotly_chart(
        make_line_chart(prices, "Adjusted Close Prices", "Price"),
        use_container_width=True
    )

    with st.expander("Preview cleaned price data"):
        st.dataframe(prices.tail(20), use_container_width=True)

with tab2:
    st.subheader("Summary Statistics")
    stats_df = summary_stats(all_returns)
    st.dataframe(stats_df.style.format({
        "Annualized Mean Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Skewness": "{:.3f}",
        "Kurtosis": "{:.3f}",
        "Min Daily Return": "{:.2%}",
        "Max Daily Return": "{:.2%}",
    }), use_container_width=True)

    st.subheader("Cumulative Wealth Index")
    wealth_selection = st.multiselect(
        "Choose series to display",
        options=all_returns.columns.tolist(),
        default=all_returns.columns.tolist(),
    )

    wealth_df = (1 + all_returns[wealth_selection]).cumprod() * 10000 if wealth_selection else pd.DataFrame()

    if wealth_df.empty:
        st.info("Select at least one series to display the cumulative wealth chart.")
    else:
        st.plotly_chart(
            make_line_chart(wealth_df, "Growth of $10,000", "Portfolio Value ($)"),
            use_container_width=True
        )

    st.subheader("Distribution Analysis")
    selected_dist_stock = st.selectbox("Select a stock for distribution analysis", asset_cols)
    dist_view = st.radio(
        "Select plot type",
        options=["Histogram + Normal Curve", "Q-Q Plot"],
        horizontal=True
    )

    stock_ret = asset_returns[selected_dist_stock].dropna()

    if dist_view == "Histogram + Normal Curve":
        mu = stock_ret.mean()
        sigma = stock_ret.std()
        x_vals = np.linspace(stock_ret.min(), stock_ret.max(), 300)
        pdf_vals = stats.norm.pdf(x_vals, mu, sigma)

        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=stock_ret,
            histnorm="probability density",
            name="Daily Returns",
            opacity=0.75
        ))
        hist_fig.add_trace(go.Scatter(
            x=x_vals,
            y=pdf_vals,
            mode="lines",
            name="Fitted Normal Curve"
        ))
        hist_fig.update_layout(
            title=f"{selected_dist_stock}: Histogram of Daily Returns with Normal Overlay",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        qq = stats.probplot(stock_ret, dist="norm")
        theoretical = qq[0][0]
        ordered = qq[0][1]
        slope = qq[1][0]
        intercept = qq[1][1]

        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(
            x=theoretical,
            y=ordered,
            mode="markers",
            name="Observed Quantiles"
        ))
        qq_fig.add_trace(go.Scatter(
            x=theoretical,
            y=slope * theoretical + intercept,
            mode="lines",
            name="Reference Line"
        ))
        qq_fig.update_layout(
            title=f"{selected_dist_stock}: Q-Q Plot vs Normal",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Ordered Returns",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(qq_fig, use_container_width=True)

with tab3:
    st.subheader("Rolling Annualized Volatility")
    rolling_vol_df = asset_returns.rolling(rolling_vol_window).std() * np.sqrt(TRADING_DAYS)
    st.plotly_chart(
        make_line_chart(
            rolling_vol_df,
            f"Rolling Annualized Volatility ({rolling_vol_window}-Day Window)",
            "Annualized Volatility"
        ),
        use_container_width=True
    )

    st.subheader("Drawdown Analysis")
    drawdown_stock = st.selectbox("Select a stock for drawdown analysis", asset_cols, key="drawdown_stock")
    drawdown_wealth = (1 + asset_returns[drawdown_stock]).cumprod()
    drawdown_peak = drawdown_wealth.cummax()
    drawdown_series = drawdown_wealth / drawdown_peak - 1

    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.metric("Maximum Drawdown", format_pct(drawdown_series.min()))
    with col_b:
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, mode="lines", name=drawdown_stock))
        dd_fig.update_layout(
            title=f"{drawdown_stock} Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(dd_fig, use_container_width=True)

    st.subheader("Risk-Adjusted Metrics")
    risk_df = risk_adjusted_table(all_returns, rf_annual)
    st.dataframe(risk_df.style.format({
        "Sharpe Ratio": "{:.3f}",
        "Sortino Ratio": "{:.3f}",
    }), use_container_width=True)

with tab4:
    st.subheader("Correlation Heatmap")
    corr_df = asset_returns.corr()

    heatmap_fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Pairwise Correlation Matrix of Daily Returns"
    )
    heatmap_fig.update_layout(height=550)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.subheader("Rolling Correlation")
    c1, c2 = st.columns(2)
    with c1:
        stock_1 = st.selectbox("Stock 1", asset_cols, key="rollcorr_s1")
    with c2:
        stock_2_choices = [s for s in asset_cols if s != stock_1]
        stock_2 = st.selectbox("Stock 2", stock_2_choices, key="rollcorr_s2")

    rolling_corr = asset_returns[stock_1].rolling(rolling_corr_window).corr(asset_returns[stock_2])

    rc_fig = go.Figure()
    rc_fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name=f"{stock_1} vs {stock_2}"))
    rc_fig.update_layout(
        title=f"Rolling Correlation: {stock_1} vs {stock_2} ({rolling_corr_window}-Day Window)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(rc_fig, use_container_width=True)

    with st.expander("Show covariance matrix"):
        cov_df = asset_returns.cov()
        st.dataframe(cov_df, use_container_width=True)

with tab5:
    st.subheader("Equal-Weight Portfolio")
    eq_display = pd.DataFrame({
        "Metric": [
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
        ],
        "Value": [
            eq_metrics["Annualized Return"],
            eq_metrics["Annualized Volatility"],
            eq_metrics["Sharpe Ratio"],
            eq_metrics["Sortino Ratio"],
            eq_metrics["Max Drawdown"],
        ]
    })
    st.dataframe(eq_display.style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)

    st.subheader("Optimized Portfolios")

    gmv_weights_series = pd.Series(gmv_weights, index=asset_cols, name="GMV Weight")
    tan_weights_series = pd.Series(tan_weights, index=asset_cols, name="Tangency Weight")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Global Minimum Variance (GMV) Portfolio**")
        gmv_table = pd.DataFrame({
            "Metric": [
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown",
            ],
            "Value": [
                gmv_metrics["Annualized Return"],
                gmv_metrics["Annualized Volatility"],
                gmv_metrics["Sharpe Ratio"],
                gmv_metrics["Sortino Ratio"],
                gmv_metrics["Max Drawdown"],
            ]
        })
        st.dataframe(gmv_table.style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.plotly_chart(
            make_bar_chart(gmv_weights_series, "GMV Portfolio Weights", "Weight"),
            use_container_width=True
        )

    with col2:
        st.markdown("**Maximum Sharpe (Tangency) Portfolio**")
        tan_table = pd.DataFrame({
            "Metric": [
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown",
            ],
            "Value": [
                tan_metrics["Annualized Return"],
                tan_metrics["Annualized Volatility"],
                tan_metrics["Sharpe Ratio"],
                tan_metrics["Sortino Ratio"],
                tan_metrics["Max Drawdown"],
            ]
        })
        st.dataframe(tan_table.style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.plotly_chart(
            make_bar_chart(tan_weights_series, "Tangency Portfolio Weights", "Weight"),
            use_container_width=True
        )

    st.subheader("Risk Contribution (Percentage Risk Contribution, PRC)")
    st.caption(
        "PRC shows how much of total portfolio volatility comes from each position. "
        "A stock can have a smaller weight than another stock but still contribute more risk if it is more volatile or more correlated with the rest of the portfolio."
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        st.plotly_chart(
            make_bar_chart(gmv_prc, "GMV Portfolio Risk Contribution", "PRC"),
            use_container_width=True
        )
    with rc2:
        st.plotly_chart(
            make_bar_chart(tan_prc, "Tangency Portfolio Risk Contribution", "PRC"),
            use_container_width=True
        )

    st.subheader("Custom Portfolio Builder")
    st.caption("Sliders are normalized so the weights sum to 100%.")

    raw_slider_weights = []
    custom_cols = st.columns(min(4, n_assets))
    for i, ticker in enumerate(asset_cols):
        with custom_cols[i % len(custom_cols)]:
            raw_val = st.slider(
                f"{ticker}",
                min_value=0.0,
                max_value=100.0,
                value=float(round(100 / n_assets, 2)),
                step=1.0,
                key=f"custom_{ticker}"
            )
            raw_slider_weights.append(raw_val)

    raw_slider_weights = np.array(raw_slider_weights, dtype=float)
    if raw_slider_weights.sum() == 0:
        custom_weights = np.ones(n_assets) / n_assets
        st.warning("All slider values were zero, so equal weights were used.")
    else:
        custom_weights = raw_slider_weights / raw_slider_weights.sum()

    custom_weights_series = pd.Series(custom_weights, index=asset_cols, name="Custom Weight")
    custom_metrics = build_portfolio_metrics(custom_weights, asset_returns, rf_annual)

    st.markdown("**Normalized Custom Weights**")
    st.dataframe(
        pd.DataFrame({"Weight": custom_weights_series}).style.format({"Weight": "{:.2%}"}),
        use_container_width=True
    )

    custom_metrics_df = pd.DataFrame({
        "Metric": [
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
        ],
        "Value": [
            custom_metrics["Annualized Return"],
            custom_metrics["Annualized Volatility"],
            custom_metrics["Sharpe Ratio"],
            custom_metrics["Sortino Ratio"],
            custom_metrics["Max Drawdown"],
        ]
    })
    st.dataframe(custom_metrics_df.style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)

    st.plotly_chart(
        make_bar_chart(custom_weights_series, "Custom Portfolio Weights", "Weight"),
        use_container_width=True
    )

    st.subheader("Efficient Frontier")
    st.caption(
        "The efficient frontier shows the highest expected return available for each level of volatility under the no-short-selling constraint. "
        "The Capital Allocation Line (CAL) starts at the risk-free rate and passes through the tangency portfolio, representing the best risk-return tradeoff when mixing the tangency portfolio with the risk-free asset."
    )

    frontier_fig = go.Figure()

    if not frontier_df.empty:
        frontier_fig.add_trace(go.Scatter(
            x=frontier_df["Volatility"],
            y=frontier_df["Target Return"],
            mode="lines",
            name="Efficient Frontier"
        ))

    asset_points_x = asset_returns.std().values * np.sqrt(TRADING_DAYS)
    asset_points_y = asset_returns.mean().values * TRADING_DAYS
    frontier_fig.add_trace(go.Scatter(
        x=asset_points_x,
        y=asset_points_y,
        mode="markers+text",
        text=asset_cols,
        textposition="top center",
        name="Individual Stocks"
    ))

    bm_x = annualized_vol(benchmark_returns)
    bm_y = annualized_return(benchmark_returns)
    frontier_fig.add_trace(go.Scatter(
        x=[bm_x], y=[bm_y],
        mode="markers+text",
        text=[BENCHMARK],
        textposition="top center",
        name="S&P 500 Benchmark",
        marker=dict(size=10)
    ))

    frontier_fig.add_trace(go.Scatter(
        x=[eq_metrics["Annualized Volatility"]],
        y=[eq_metrics["Annualized Return"]],
        mode="markers+text",
        text=["Equal Weight"],
        textposition="bottom center",
        name="Equal Weight",
        marker=dict(size=11)
    ))

    frontier_fig.add_trace(go.Scatter(
        x=[gmv_metrics["Annualized Volatility"]],
        y=[gmv_metrics["Annualized Return"]],
        mode="markers+text",
        text=["GMV"],
        textposition="bottom center",
        name="GMV",
        marker=dict(size=11)
    ))

    frontier_fig.add_trace(go.Scatter(
        x=[tan_metrics["Annualized Volatility"]],
        y=[tan_metrics["Annualized Return"]],
        mode="markers+text",
        text=["Tangency"],
        textposition="bottom center",
        name="Tangency",
        marker=dict(size=11)
    ))

    frontier_fig.add_trace(go.Scatter(
        x=[custom_metrics["Annualized Volatility"]],
        y=[custom_metrics["Annualized Return"]],
        mode="markers+text",
        text=["Custom"],
        textposition="bottom center",
        name="Custom",
        marker=dict(size=11)
    ))

    tan_vol = tan_metrics["Annualized Volatility"]
    tan_ret = tan_metrics["Annualized Return"]
    max_x = tan_vol * 1.15
    if not frontier_df.empty:
        max_x = max(max_x, frontier_df["Volatility"].max() * 1.15)

    cal_x = np.linspace(0, max_x, 50)
    cal_slope = (tan_ret - rf_annual) / tan_vol if tan_vol != 0 else 0
    cal_y = rf_annual + cal_slope * cal_x

    frontier_fig.add_trace(go.Scatter(
        x=cal_x,
        y=cal_y,
        mode="lines",
        name="Capital Allocation Line"
    ))

    frontier_fig.update_layout(
        title="Efficient Frontier and Capital Allocation Line",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white",
        height=600,
    )
    st.plotly_chart(frontier_fig, use_container_width=True)

    st.subheader("Portfolio Wealth Comparison")
    comparison_wealth = pd.DataFrame({
        "Equal Weight": (1 + eq_metrics["Daily Returns"]).cumprod() * 10000,
        "GMV": (1 + gmv_metrics["Daily Returns"]).cumprod() * 10000,
        "Tangency": (1 + tan_metrics["Daily Returns"]).cumprod() * 10000,
        "Custom": (1 + custom_metrics["Daily Returns"]).cumprod() * 10000,
        "S&P 500": (1 + benchmark_returns).cumprod() * 10000,
    })

    st.plotly_chart(
        make_line_chart(
            comparison_wealth,
            "Growth of $10,000: Portfolio Comparison",
            "Portfolio Value ($)"
        ),
        use_container_width=True
    )

    st.subheader("Portfolio Comparison Table")
    comparison_df = pd.DataFrame({
        "Annualized Return": [
            eq_metrics["Annualized Return"],
            gmv_metrics["Annualized Return"],
            tan_metrics["Annualized Return"],
            custom_metrics["Annualized Return"],
            annualized_return(benchmark_returns),
        ],
        "Annualized Volatility": [
            eq_metrics["Annualized Volatility"],
            gmv_metrics["Annualized Volatility"],
            tan_metrics["Annualized Volatility"],
            custom_metrics["Annualized Volatility"],
            annualized_vol(benchmark_returns),
        ],
        "Sharpe Ratio": [
            eq_metrics["Sharpe Ratio"],
            gmv_metrics["Sharpe Ratio"],
            tan_metrics["Sharpe Ratio"],
            custom_metrics["Sharpe Ratio"],
            sharpe_ratio(benchmark_returns, rf_annual),
        ],
        "Sortino Ratio": [
            eq_metrics["Sortino Ratio"],
            gmv_metrics["Sortino Ratio"],
            tan_metrics["Sortino Ratio"],
            custom_metrics["Sortino Ratio"],
            sortino_ratio(benchmark_returns, rf_annual),
        ],
        "Max Drawdown": [
            eq_metrics["Max Drawdown"],
            gmv_metrics["Max Drawdown"],
            tan_metrics["Max Drawdown"],
            custom_metrics["Max Drawdown"],
            max_drawdown_from_returns(benchmark_returns),
        ],
    }, index=["Equal Weight", "GMV", "Tangency", "Custom", "S&P 500"])

    st.dataframe(comparison_df.style.format({
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Sortino Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}",
    }), use_container_width=True)

with tab6:
    st.subheader("Estimation Window Sensitivity")
    st.caption(
        "Mean-variance optimization is sensitive to the historical inputs used to estimate returns and covariances. "
        "This section shows how GMV and tangency allocations can change when the lookback window changes."
    )

    st.dataframe(
        sensitivity_df.style.format({
            "Annualized Return": "{:.2%}",
            "Annualized Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.3f}",
        }),
        use_container_width=True
    )

    if not weights_long.empty:
        chosen_portfolio = st.radio(
            "Choose portfolio weights to compare across windows",
            ["GMV", "Tangency"],
            horizontal=True
        )

        chart_df = weights_long[weights_long["Portfolio"] == chosen_portfolio]

        sens_fig = px.bar(
            chart_df,
            x="Window",
            y="Weight",
            color="Ticker",
            barmode="group",
            title=f"{chosen_portfolio} Portfolio Weights Across Estimation Windows"
        )
        sens_fig.update_layout(
            xaxis_title="Estimation Window",
            yaxis_title="Weight",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(sens_fig, use_container_width=True)

with st.expander("About / Methodology"):
    st.markdown(
        """
**Analytical Methods**
- Uses daily simple returns, not log returns.
- Annualized return = mean daily return × 252.
- Annualized volatility = daily standard deviation × √252.
- Sharpe and Sortino ratios use the user-specified annualized risk-free rate.
- Downside deviation for Sortino uses only returns below the daily risk-free threshold.
- Cumulative wealth uses `(1 + r).cumprod()`.

**Portfolio Optimization**
- Optimizations use `scipy.optimize.minimize` with:
  - no-short-selling bounds `(0, 1)`
  - full-investment constraint `sum(weights) = 1`
- GMV minimizes portfolio volatility.
- Tangency maximizes Sharpe ratio.
- Efficient frontier is computed by solving a constrained optimization for each target return.

**Risk Contribution**
- Percentage risk contribution is computed as:
  - `PRC_i = w_i * (Σw)_i / σ_p²`
- PRCs sum to approximately 1.

**Data Source**
- Data comes from Yahoo Finance via `yfinance`.
- The app uses adjusted closing prices when available.
        """
    )