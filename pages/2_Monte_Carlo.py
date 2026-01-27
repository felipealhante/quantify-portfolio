import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.title("Monte Carlo Simulation")
st.caption("Simulates future prices using historical daily log returns.")

# ---------------------------
# Helpers
# ---------------------------
def clean_ticker(t: str) -> str:
    return t.strip().upper().replace("ˆ", "^")

@st.cache_data(show_spinner=False)
def download_price(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)

    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.astype(float)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Inputs")

ticker = clean_ticker(st.sidebar.text_input("Ticker"))

years = st.sidebar.slider("Years of history", 1, 15, 5)
horizon_days = st.sidebar.slider("Forecast horizon (trading days)", 5, 2000, 252)

# MAX 400 simulations
n_sims = st.sidebar.slider("Simulations (max 400)", 50, 400, 300, step=10)

seed = st.sidebar.number_input("Random seed", 0, 10_000_000, 42)

run = st.sidebar.button("Run simulation", type="primary", use_container_width=True)
if not run:
    st.stop()

# ---------------------------
# Download history
# ---------------------------
today = dt.date.today()
start = today - dt.timedelta(days=365 * years)

with st.spinner("Downloading price history..."):
    prices = download_price(ticker, start, today)

if prices.empty or len(prices) < 50:
    st.error("No usable data returned for this ticker.")
    st.stop()

# ---------------------------
# Returns stats
# ---------------------------
logret = np.log(prices / prices.shift(1)).dropna()
mu = float(logret.mean())
sigma = float(logret.std(ddof=0))

S0 = float(prices.iloc[-1])

c1, c2, c3 = st.columns(3)
c1.metric("Last Price", f"{S0:,.2f}")
c2.metric("Daily μ", f"{mu:.6f}")
c3.metric("Daily σ", f"{sigma:.6f}")

# ---------------------------
# Monte Carlo Simulation
# ---------------------------
np.random.seed(int(seed))

T = int(horizon_days)
N = int(n_sims)

Z = np.random.normal(size=(N, T))

drift = mu - 0.5 * sigma**2
increments = drift + sigma * Z

paths = S0 * np.exp(np.cumsum(increments, axis=1))
paths = np.concatenate([np.full((N, 1), S0), paths], axis=1)

ending = paths[:, -1]
p5, p50, p95 = np.percentile(ending, [5, 50, 95])

# ---------------------------
# Key result
# ---------------------------
st.subheader("Final Price Statistics")

m1, m2, m3 = st.columns(3)
m1.metric("5th Percentile", f"{p5:,.2f}")
m2.metric("Median Final Price", f"{p50:,.2f}")
m3.metric("95th Percentile", f"{p95:,.2f}")

st.caption(f"Horizon: {T} trading days | Simulations: {N}")

# ---------------------------
# Plot 1: Paths
# ---------------------------
st.subheader("Simulated Price Paths")

t_index = np.arange(paths.shape[1])

fig_paths = go.Figure()

# Plot all paths (<= 400)
for i in range(paths.shape[0]):
    fig_paths.add_trace(go.Scatter(
        x=t_index,
        y=paths[i],
        mode="lines",
        opacity=0.25,
        line=dict(width=1),
        showlegend=False
    ))

# Percentile bands over time
fig_paths.add_trace(go.Scatter(
    x=t_index, y=np.percentile(paths, 50, axis=0),
    name="Median", line=dict(width=3)
))
fig_paths.add_trace(go.Scatter(
    x=t_index, y=np.percentile(paths, 5, axis=0),
    name="5th pct", line=dict(dash="dash")
))
fig_paths.add_trace(go.Scatter(
    x=t_index, y=np.percentile(paths, 95, axis=0),
    name="95th pct", line=dict(dash="dash")
))

fig_paths.update_layout(
    template="plotly_white",
    height=520,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="Days Ahead",
    yaxis_title="Simulated Price"
)

st.plotly_chart(fig_paths, use_container_width=True)

# ---------------------------
# Plot 2: Distribution
# ---------------------------
st.subheader("Distribution of Final Prices")

end_df = pd.DataFrame({"Final Price": ending})
fig_hist = px.histogram(end_df, x="Final Price", nbins=40, template="plotly_white")

# Median line (explicitly demonstrated)
fig_hist.add_vline(
    x=p50,
    line_width=3,
    line_dash="dash",
    line_color="black",
    annotation_text="Median",
    annotation_position="top right",
)

fig_hist.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------
# Debug
# ---------------------------
with st.expander("Debug"):
    st.write("Ticker:", ticker)
    st.write("Price range:", prices.index.min().date(), "→", prices.index.max().date())
    st.dataframe(prices.tail(10), use_container_width=True)

