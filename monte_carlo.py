import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Monte Carlo Portfolio Simulator", layout="wide")
st.title("Monte Carlo Portfolio Simulator (Yahoo Finance)")
st.caption("Simulate future portfolio values using historical daily log-returns (GBM-style simulation).")

# ----------------------------
# Helpers (robust for AAPL/NVDA + multi-ticker)
# ----------------------------
def normalize_tickers(raw: str) -> list[str]:
    raw = raw.replace("\n", ",").replace(" ", ",")
    parts = [t.strip().upper().replace("ˆ", "^") for t in raw.split(",") if t.strip()]
    seen, out = set(), []
    for t in parts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


@st.cache_data(show_spinner=False)
def download_close(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    auto_adjust=True -> adjusted close stored in 'Close'
    Handles single vs multi ticker output reliably.
    Returns DataFrame columns=tickers
    """
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Multi ticker -> MultiIndex columns like ('Close','AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()
        close = df["Close"].copy()
        close.columns = [str(c).upper() for c in close.columns]
    else:
        # Single ticker -> normal columns include 'Close'
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df[["Close"]].copy()
        close.columns = [tickers[0].upper()]

    close.index = pd.to_datetime(close.index)
    close = close.sort_index().dropna(how="all").ffill()
    return close


def ensure_positive_definite(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Numerical stability for covariance (sometimes near-singular).
    """
    cov = np.asarray(cov, dtype=float)
    # Symmetrize
    cov = 0.5 * (cov + cov.T)
    # Add small jitter to diagonal
    cov += np.eye(cov.shape[0]) * eps
    return cov


def format_money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

# ----------------------------
# UI
# ----------------------------
tickers_raw = st.text_area("Tickers (comma-separated)", value="AAPL, MSFT, NVDA", height=70)
tickers = normalize_tickers(tickers_raw)

today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 3)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("History start date", value=default_start)
with c2:
    end_date = st.date_input("History end date", value=today)

st.markdown("### Portfolio holdings (optional)")
st.write("If you enter shares, the simulation runs in **dollar values**. If you leave shares as 1 for all, it’s just equal-share portfolio.")

if not tickers:
    st.info("Add at least one ticker.")
    st.stop()

hold_df = pd.DataFrame({"Ticker": tickers, "Shares": [1.0] * len(tickers)})
holdings = st.data_editor(
    hold_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Shares": st.column_config.NumberColumn(min_value=0.0, step=1.0),
    },
)

st.markdown("### Simulation settings")
s1, s2, s3, s4 = st.columns(4)
with s1:
    n_sims = st.number_input("Simulations", min_value=100, max_value=50000, value=5000, step=100)
with s2:
    horizon_days = st.number_input("Horizon (trading days)", min_value=5, max_value=2000, value=252, step=5)
with s3:
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)
with s4:
    method = st.selectbox("Method", ["Multivariate (correlated)", "Independent"], index=0)

rf_annual = st.number_input("Risk-free rate (annual, optional)", value=0.00, step=0.01, format="%.2f")

run = st.button("Run Monte Carlo", type="primary", use_container_width=True)
if not run:
    st.stop()

# ----------------------------
# Validate inputs
# ----------------------------
if start_date >= end_date:
    st.error("History start date must be before end date.")
    st.stop()

holdings = holdings.copy()
holdings["Ticker"] = holdings["Ticker"].astype(str).str.upper()
holdings["Shares"] = pd.to_numeric(holdings["Shares"], errors="coerce").fillna(1.0)
holdings.loc[holdings["Shares"] <= 0, "Shares"] = 1.0

# ----------------------------
# Download history
# ----------------------------
with st.spinner("Downloading historical prices..."):
    prices = download_close(holdings["Ticker"].tolist(), start_date, end_date)

if prices.empty:
    st.error("No data returned. Check tickers/date range.")
    st.stop()

# Drop tickers with no data
prices = prices.dropna(axis=1, how="all").ffill()

available = list(prices.columns)
missing = [t for t in holdings["Ticker"] if t not in available]
if missing:
    st.warning(f"These tickers returned no data and were skipped: {', '.join(missing)}")

prices = prices.dropna(how="any")  # align common dates
if prices.shape[1] == 0 or len(prices) < 50:
    st.error("Not enough usable data after alignment. Try more history or fewer tickers.")
    st.stop()

# align holdings with prices columns
holdings = holdings.set_index("Ticker").reindex(prices.columns)
shares = holdings["Shares"].astype(float).values

# Current portfolio value
last_prices = prices.iloc[-1].values.astype(float)
current_value = float(np.sum(last_prices * shares))

# returns
logrets = np.log(prices / prices.shift(1)).dropna()
mu = logrets.mean().values.astype(float)          # daily mean log return per asset
cov = logrets.cov().values.astype(float)          # daily covariance matrix
cov = ensure_positive_definite(cov)

# Optionally add risk-free drift (very light-touch):
# Convert annual rf to daily and add equally to drift (approx)
rf_daily = (1.0 + float(rf_annual)) ** (1/252) - 1.0
# we’ll keep mu as empirical; rf is just shown for reference / can be used later

# ----------------------------
# Monte Carlo simulation
# ----------------------------
np.random.seed(int(seed))
n_assets = prices.shape[1]
T = int(horizon_days)
N = int(n_sims)

# Generate random shocks
if method == "Multivariate (correlated)":
    shocks = np.random.multivariate_normal(mean=np.zeros(n_assets), cov=cov, size=(N, T))
else:
    # independent using each asset variance
    std = np.sqrt(np.diag(cov))
    shocks = np.random.normal(loc=0.0, scale=std, size=(N, T, n_assets))

# Build simulated log returns: r_t = mu + shock
# Shapes:
# correlated: shocks (N,T,n_assets) after expand
if method == "Multivariate (correlated)":
    shocks = shocks.reshape(N, T, n_assets)

sim_logrets = mu.reshape(1, 1, n_assets) + shocks  # (N,T,n_assets)

# Convert to simulated prices paths per asset
# Start from last observed prices
start_prices = last_prices.reshape(1, 1, n_assets)  # (1,1,A)

# cumulative sum over time of log returns -> log price ratio
cum = np.cumsum(sim_logrets, axis=1)  # (N,T,A)
sim_prices = start_prices * np.exp(cum)  # (N,T,A)

# Portfolio value per path over time
port_values = np.sum(sim_prices * shares.reshape(1, 1, n_assets), axis=2)  # (N,T)
port_values = np.concatenate([np.full((N, 1), current_value), port_values], axis=1)  # include t=0
t_index = np.arange(port_values.shape[1])

ending_values = port_values[:, -1]
ending_return = ending_values / current_value - 1.0

# Risk stats
p5 = float(np.percentile(ending_values, 5))
p50 = float(np.percentile(ending_values, 50))
p95 = float(np.percentile(ending_values, 95))

var_95_loss = float(np.percentile(current_value - ending_values, 95))  # 95% VaR (loss)
cvar_95_loss = float(np.mean((current_value - ending_values)[(current_value - ending_values) >= var_95_loss]))

# ----------------------------
# Display summary
# ----------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Portfolio Value", format_money(current_value))
m2.metric("Median Ending Value", format_money(p50))
m3.metric("5th–95th Ending Range", f"{format_money(p5)}  →  {format_money(p95)}")
m4.metric("Median Return", f"{(p50/current_value - 1)*100:.2f}%")

st.caption(f"VaR(95%) loss: {format_money(var_95_loss)} | CVaR(95%) loss: {format_money(cvar_95_loss)}")

# ----------------------------
# Plot 1: Sample paths
# ----------------------------
st.subheader("Simulated Portfolio Value Paths")

# Downsample paths to plot (so browser doesn’t die)
max_paths_to_plot = 200
idx = np.linspace(0, N - 1, min(N, max_paths_to_plot)).astype(int)
paths_plot = port_values[idx, :]

fig_paths = go.Figure()
for i in range(paths_plot.shape[0]):
    fig_paths.add_trace(go.Scatter(
        x=t_index,
        y=paths_plot[i],
        mode="lines",
        line=dict(width=1),
        opacity=0.25,
        showlegend=False
    ))

# Add percentile bands
pctl5 = np.percentile(port_values, 5, axis=0)
pctl50 = np.percentile(port_values, 50, axis=0)
pctl95 = np.percentile(port_values, 95, axis=0)

fig_paths.add_trace(go.Scatter(x=t_index, y=pctl50, mode="lines", name="Median", line=dict(width=3)))
fig_paths.add_trace(go.Scatter(x=t_index, y=pctl5, mode="lines", name="5th pct", line=dict(dash="dash")))
fig_paths.add_trace(go.Scatter(x=t_index, y=pctl95, mode="lines", name="95th pct", line=dict(dash="dash")))

fig_paths.update_layout(
    template="plotly_white",
    height=520,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Days ahead",
    yaxis_title="Portfolio Value ($)",
)
st.plotly_chart(fig_paths, use_container_width=True)

# ----------------------------
# Plot 2: Distribution of ending values
# ----------------------------
st.subheader("Distribution of Ending Portfolio Values")

end_df = pd.DataFrame({"Ending Value ($)": ending_values})
fig_hist = px.histogram(end_df, x="Ending Value ($)", nbins=60, template="plotly_white")
fig_hist.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=20))
st.plotly_chart(fig_hist, use_container_width=True)

# ----------------------------
# Table: key percentiles
# ----------------------------
st.subheader("Percentiles")
pct_table = pd.DataFrame(
    {
        "Percentile": ["5%", "50% (Median)", "95%"],
        "Ending Value": [p5, p50, p95],
        "Return vs Now": [(p5/current_value - 1), (p50/current_value - 1), (p95/current_value - 1)],
    }
)
pct_table["Ending Value"] = pct_table["Ending Value"].map(format_money)
pct_table["Return vs Now"] = pct_table["Return vs Now"].map(lambda x: f"{x*100:.2f}%")
st.dataframe(pct_table, use_container_width=True)

with st.expander("Debug (ticker/data checks)"):
    st.write("Tickers used:", list(prices.columns))
    st.write("Price date range:", prices.index.min(), "→", prices.index.max())
    st.write("Last prices:", dict(zip(prices.columns, last_prices)))
