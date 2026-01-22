import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Monte Carlo (Single Ticker)", layout="wide")
st.title("Monte Carlo Simulation (Single Ticker)")
st.caption("Simulates future prices using historical daily log returns (GBM-style).")

# ---------------------------
# Robust ticker + data download
# ---------------------------
def clean_ticker(t: str) -> str:
    # fixes mac weird caret: ˆ vs ^
    return t.strip().upper().replace("ˆ", "^")

@st.cache_data(show_spinner=False)
def download_price(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    """
    Uses auto_adjust=True so 'Close' is adjusted close.
    Returns a clean Series of prices.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if "Close" not in df.columns:
        return pd.Series(dtype=float)

    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().dropna()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s.name = ticker
    return s.astype(float)

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Inputs")

ticker = clean_ticker(st.sidebar.text_input("Ticker", "AAPL"))

today = dt.date.today()
years = st.sidebar.slider("Years of history", 1, 15, 5)
start = today - dt.timedelta(days=365 * years)
end = today

initial_price_mode = st.sidebar.selectbox(
    "Starting price",
    ["Use latest market close", "Custom"],
    index=0
)
custom_start_price = st.sidebar.number_input("Custom start price", min_value=0.0, value=100.0, step=1.0)

horizon_days = st.sidebar.slider("Forecast horizon (trading days)", 5, 2000, 252)
n_sims = st.sidebar.slider("Number of simulations", 100, 20000, 3000, step=100)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

run = st.sidebar.button("Run simulation", type="primary", use_container_width=True)
if not run:
    st.stop()

# ---------------------------
# Download history
# ---------------------------
with st.spinner("Downloading price history..."):
    prices = download_price(ticker, start, end)

if prices.empty or len(prices) < 50:
    st.error(
        "No usable data returned. Common causes:\n"
        "- Ticker typo\n"
        "- Yahoo blocked the request temporarily\n"
        "- Too little price history\n\n"
        "Try again, or try a different ticker (e.g., AAPL, NVDA)."
    )
    st.stop()

# ---------------------------
# Compute returns
# ---------------------------
logret = np.log(prices / prices.shift(1)).dropna()
mu = float(logret.mean())     # daily mean log return
sigma = float(logret.std(ddof=0))  # daily std dev

last_price = float(prices.iloc[-1])
S0 = last_price if initial_price_mode == "Use latest market close" else float(custom_start_price)

c1, c2, c3 = st.columns(3)
c1.metric("Last price", f"{last_price:,.2f}")
c2.metric("Daily μ (log)", f"{mu:.6f}")
c3.metric("Daily σ (log)", f"{sigma:.6f}")

# ---------------------------
# Monte Carlo simulation (GBM)
# ---------------------------
np.random.seed(int(seed))
T = int(horizon_days)
N = int(n_sims)

# Z ~ N(0,1) shape (N,T)
Z = np.random.normal(size=(N, T))

# log-price increments: (mu - 0.5*sigma^2) + sigma*Z
drift = (mu - 0.5 * sigma**2)
increments = drift + sigma * Z

# build paths: S_t = S0 * exp(cumsum(increments))
paths = S0 * np.exp(np.cumsum(increments, axis=1))

# include t=0 for plotting
paths = np.concatenate([np.full((N, 1), S0), paths], axis=1)
t_index = np.arange(paths.shape[1])

ending = paths[:, -1]

p5, p50, p95 = np.percentile(ending, [5, 50, 95])
st.write(f"**Ending price percentiles** (after {T} trading days): 5%={p5:,.2f} | median={p50:,.2f} | 95%={p95:,.2f}")

# ---------------------------
# Plot: simulated paths (sample) + percentile lines
# ---------------------------
st.subheader("Simulated Price Paths")

max_paths_to_plot = 200
idx = np.linspace(0, N - 1, min(N, max_paths_to_plot)).astype(int)
sample_paths = paths[idx, :]

fig_paths = go.Figure()

for i in range(sample_paths.shape[0]):
    fig_paths.add_trace(go.Scatter(
        x=t_index,
        y=sample_paths[i],
        mode="lines",
        line=dict(width=1),
        opacity=0.25,
        showlegend=False
    ))

# percentile bands over time
pctl5 = np.percentile(paths, 5, axis=0)
pctl50 = np.percentile(paths, 50, axis=0)
pctl95 = np.percentile(paths, 95, axis=0)

fig_paths.add_trace(go.Scatter(x=t_index, y=pctl50, mode="lines", name="Median", line=dict(width=3)))
fig_paths.add_trace(go.Scatter(x=t_index, y=pctl5, mode="lines", name="5th pct", line=dict(dash="dash")))
fig_paths.add_trace(go.Scatter(x=t_index, y=pctl95, mode="lines", name="95th pct", line=dict(dash="dash")))

fig_paths.update_layout(
    template="plotly_white",
    height=520,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Days ahead",
    yaxis_title="Simulated Price",
)
st.plotly_chart(fig_paths, use_container_width=True)

# ---------------------------
# Plot: distribution of ending prices
# ---------------------------
st.subheader("Distribution of Ending Prices")

end_df = pd.DataFrame({"Ending Price": ending})
fig_hist = px.histogram(end_df, x="Ending Price", nbins=60, template="plotly_white")
fig_hist.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=20))
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------
# Debug: show last few prices
# ---------------------------
with st.expander("Debug (download check)"):
    st.write("Ticker used:", ticker)
    st.write("Date range:", prices.index.min().date(), "→", prices.index.max().date())
    st.dataframe(prices.tail(10), use_container_width=True)
