
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")

# ---------- Title (same vibe as your screenshot)
st.markdown(
    "<h1 style='text-align:center; font-size:64px; margin-bottom:10px;'>Monte Carlo Simulator</h1>",
    unsafe_allow_html=True,
)

# ---------- Inputs (center column for the clean look)
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

    run = st.button("Run simulation", type="primary", use_container_width=True)

if not run:
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

@st.cache_data(show_spinner=False)
def download_prices(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)
    return data["Close"].dropna()

prices = download_prices(ticker, start_date, end_date)

if prices.empty:
    st.error(f"No valid data for ticker '{ticker}'. Try another symbol or date range.")
    st.stop()

if len(prices) < 10:
    st.error("Not enough data points to run Monte Carlo (need at least ~10 closes).")
    st.stop()

# ---------- Log returns (GBM-ish like your code)
log_returns = np.log(prices / prices.shift(1)).dropna()
mu = float(log_returns.mean())
sigma = float(log_returns.std())
last_price = float(prices.iloc[-1])

# ---------- Monte Carlo simulation
rng = np.random.default_rng(int(seed))
sim_paths = np.zeros((num_days + 1, int(num_simulations)), dtype=float)
sim_paths[0, :] = last_price

for j in range(int(num_simulations)):
    for t in range(1, num_days + 1):
        z = rng.normal(0, 1)
        sim_paths[t, j] = sim_paths[t - 1, j] * np.exp(mu + sigma * z)

# ---------- Percentiles
p5 = np.percentile(sim_paths, 5, axis=1)
p50 = np.percentile(sim_paths, 50, axis=1)
p95 = np.percentile(sim_paths, 95, axis=1)

# Build a DataFrame for plotting
days = np.arange(0, num_days + 1)
paths_df = pd.DataFrame(sim_paths, index=days)
paths_long = paths_df.reset_index().melt(id_vars="index", var_name="path", value_name="price")
paths_long = paths_long.rename(columns={"index": "day"})

bands = pd.DataFrame({"day": days, "p5": p5, "p50": p50, "p95": p95})
final_prices = sim_paths[-1, :]

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

# ---------- Plot paths (sample a bit so it stays fast + readable)
max_to_plot = min(200, int(num_simulations))  # show at most 200 paths
paths_long_plot = paths_long[paths_long["path"].astype(int) < max_to_plot]

fig_paths = px.line(
    paths_long_plot,
    x="day",
    y="price",
    color="path",
    template="plotly_white",
)

# hide legend for the 200 paths (too messy), keep clean grid
fig_paths.update_layout(
    height=520,
    showlegend=False,
    margin=dict(l=20, r=20, t=10, b=40),
    xaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
)

# Add percentile bands/lines
fig_paths.add_scatter(x=bands["day"], y=bands["p50"], mode="lines", name="Median (50%)", line=dict(width=4))
fig_paths.add_scatter(x=bands["day"], y=bands["p5"], mode="lines", name="5%", line=dict(width=3))
fig_paths.add_scatter(x=bands["day"], y=bands["p95"], mode="lines", name="95%", line=dict(width=3))

# Now show legend only for percentile lines
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
