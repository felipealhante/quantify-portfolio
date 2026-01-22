import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.express as px

# yfinance can fail on Streamlit Cloud sometimes (network/rate limits)
# so we import it safely
try:
    import yfinance as yf
    YF_OK = True
except Exception as e:
    yf = None
    YF_OK = False
    YF_IMPORT_ERROR = str(e)

st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")

# ---------- Title
st.markdown(
    "<h1 style='text-align:center; font-size:64px; margin-bottom:10px;'>Monte Carlo Simulator</h1>",
    unsafe_allow_html=True,
)

# ---------- Inputs
colL, colC, colR = st.columns([1, 2, 1])
with colC:
    ticker = st.text_input("One Ticker (Yahoo Finance)", "AAPL").strip().upper()

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 3)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=today)

    st.markdown("---")

    num_simulations = st.slider("Number of simulations", 50, 2000, 300, step=50)
    num_days = st.slider("Forecast horizon (trading days)", 30, 756, 252, step=21)

    seed = st.number_input("Random seed", value=42, step=1)

    # IMPORTANT: fallback mode to prove Streamlit is working even if Yahoo fails
    fallback_mode = st.checkbox(
        "Fallback mode (use synthetic prices if Yahoo fails)",
        value=True
    )

    run = st.button("Run simulation", type="primary", use_container_width=True)

# Always show diagnostics so the app never looks blank
with st.expander("Diagnostics (open if blank / errors)", expanded=False):
    st.write("Python:", __import__("sys").version)
    st.write("NumPy:", np.__version__)
    st.write("Pandas:", pd.__version__)
    st.write("Plotly:", px.__version__ if hasattr(px, "__version__") else "unknown")
    st.write("yfinance import ok:", YF_OK)
    if not YF_OK:
        st.error(f"yfinance import failed: {YF_IMPORT_ERROR}")

if not run:
    st.info("Set inputs and click **Run simulation**.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

@st.cache_data(show_spinner=False, ttl=60 * 60)  # cache for 1 hour
def download_prices(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)

    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,   # sometimes safer on cloud
        )
        if data is None or data.empty or "Close" not in data.columns:
            return pd.Series(dtype=float)

        s = data["Close"].dropna()
        s.name = "Close"
        return s
    except Exception:
        # return empty so caller can fallback instead of crashing
        return pd.Series(dtype=float)

def synthetic_prices(start_price: float = 100.0, n: int = 800, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    # mild drift + volatility, just to test UI + simulation pipeline
    mu = 0.0003
    sigma = 0.015
    rets = rng.normal(mu, sigma, size=n)
    prices = start_price * np.exp(np.cumsum(rets))
    idx = pd.RangeIndex(n, name="t")
    return pd.Series(prices, index=idx, name="Close")

with st.spinner("Downloading price data..."):
    prices = download_prices(ticker, start_date, end_date)

# If Yahoo fails, don’t crash — fallback if enabled
if prices.empty:
    if fallback_mode:
        st.warning(
            "Yahoo Finance returned no data (common on Streamlit Cloud sometimes). "
            "Using **synthetic prices** so you can verify the app works."
        )
        prices = synthetic_prices(start_price=100.0, n=900, seed=int(seed))
    else:
        st.error(
            f"No valid data for ticker '{ticker}'. "
            "Try another symbol/date range, or enable fallback mode."
        )
        st.stop()

if len(prices) < 20:
    st.error("Not enough data points to run Monte Carlo (need at least ~20 closes).")
    st.stop()

# ---------- Log returns
log_returns = np.log(prices / prices.shift(1)).dropna()
mu = float(log_returns.mean())
sigma = float(log_returns.std())
last_price = float(prices.iloc[-1])

# ---------- Vectorized Monte Carlo (FAST)
# sim_paths shape: (num_days+1, num_simulations)
rng = np.random.default_rng(int(seed))

# random shocks: (num_days, num_simulations)
Z = rng.normal(0, 1, size=(int(num_days), int(num_simulations)))

# GBM step factors: exp(mu + sigma*Z)
steps = np.exp(mu + sigma * Z)

# cumulative product along days -> price paths relative to last_price
paths = np.vstack([
    np.ones((1, int(num_simulations))),
    np.cumprod(steps, axis=0)
]) * last_price

sim_paths = paths
final_prices = sim_paths[-1, :]

# ---------- Percentiles
days = np.arange(0, int(num_days) + 1)
p5 = np.percentile(sim_paths, 5, axis=1)
p50 = np.percentile(sim_paths, 50, axis=1)
p95 = np.percentile(sim_paths, 95, axis=1)
bands = pd.DataFrame({"day": days, "p5": p5, "p50": p50, "p95": p95})

# ---------- Summary
up_10 = last_price * 1.10
down_10 = last_price * 0.90

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last price", f"{last_price:,.2f}")
col2.metric("Median final", f"{np.median(final_prices):,.2f}")
col3.metric("P(final > +10%)", f"{(final_prices > up_10).mean():.2%}")
col4.metric("P(final < -10%)", f"{(final_prices < down_10).mean():.2%}")

st.markdown(
    "<h2 style='font-size:38px; margin-top:30px;'>Monte Carlo Paths</h2>",
    unsafe_allow_html=True,
)

# ---------- Plot paths (sample for speed/readability)
max_to_plot = min(200, int(num_simulations))
paths_to_plot = sim_paths[:, :max_to_plot]

paths_df = pd.DataFrame(paths_to_plot)
paths_df["day"] = days
paths_long = paths_df.melt(id_vars="day", var_name="path", value_name="price")

fig_paths = px.line(
    paths_long,
    x="day",
    y="price",
    color="path",
    template="plotly_white",
)

fig_paths.update_layout(
    height=520,
    showlegend=False,
    margin=dict(l=20, r=20, t=10, b=40),
    xaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
)

# Percentile lines
fig_paths.add_scatter(x=bands["day"], y=bands["p50"], mode="lines", name="Median (50%)", line=dict(width=4))
fig_paths.add_scatter(x=bands["day"], y=bands["p5"], mode="lines", name="5%", line=dict(width=3))
fig_paths.add_scatter(x=bands["day"], y=bands["p95"], mode="lines", name="95%", line=dict(width=3))

fig_paths.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="left",
        x=0.0,
        title=None
    ),
    showlegend=True,
)

st.plotly_chart(fig_paths, use_container_width=True)

st.markdown(
    "<h2 style='font-size:38px; margin-top:30px;'>Final Price Distribution</h2>",
    unsafe_allow_html=True,
)

hist_df = pd.DataFrame({"final_price": final_prices})
fig_hist = px.histogram(hist_df, x="final_price", nbins=30, template="plotly_white")
fig_hist.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=10, b=40),
    xaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
)
st.plotly_chart(fig_hist, use_container_width=True)
