import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------- Page setup ----------------
st.set_page_config(page_title="Trades (Long/Short)", layout="wide")
st.title("Trades (Long/Short)")
st.caption(
    "Enter tickers + position (Long/Short) + entry price + shares. "
    "App pulls prices from Yahoo Finance to compute exposure, P/L, allocation, and portfolio performance."
)

# ---------------- Helpers ----------------
def normalize_tickers(raw: str) -> list[str]:
    parts = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        t = chunk.strip().upper().replace("ˆ", "^")  # mac caret fix
        if t:
            parts.append(t)
    seen, out = set(), []
    for t in parts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


@st.cache_data(show_spinner=False)
def download_adj_close(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusted close becomes "Close"
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()
        close = df["Close"].copy()
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.dropna(how="all").ffill()
    # Ensure string columns
    close.columns = close.columns.astype(str)
    return close



@st.cache_data(show_spinner=False)
def get_latest_prices(tickers: list[str]) -> pd.Series:
    # Uses last available close from recent history
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    if not tickers:
        return pd.Series(dtype=float)

    end = dt.date.today()
    start = end - dt.timedelta(days=20)  # a bit wider window

    px_df = download_adj_close(tickers, start, end)
    if px_df.empty:
        return pd.Series(dtype=float)

    # Drop rows where ALL tickers are NaN (more correct than dropna() which drops any row with any NaN)
    px_df = px_df.dropna(how="all")
    if px_df.empty:
        return pd.Series(dtype=float)

    last = px_df.iloc[-1].dropna()
    if last.empty:
        return pd.Series(dtype=float)

    last.index = last.index.astype(str)
    return last.astype(float)



# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.header("Inputs")

    tickers_raw = st.text_area("Tickers (comma-separated)", height=80)
    tickers = normalize_tickers(tickers_raw)

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365)

    start_date = st.date_input("Performance start date", value=default_start)
    end_date = st.date_input("Performance end date", value=today)

    st.divider()
    st.header("Options")
    show_exposure_pie = st.checkbox("Show exposure pie chart", value=True)
    show_debug = st.checkbox("Show debug details", value=False)

# Basic validations
if not tickers:
    st.info("Add at least one ticker.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ---------------- Trade inputs table ----------------
st.markdown("### Trade inputs")
st.write(
    "Enter **Position** (Long/Short), **Entry Price** (buy for Long, short-sell for Short), and **Shares**. "
    "If Shares is blank/0, it defaults to 1."
)

base_df = pd.DataFrame(
    {
        "Ticker": tickers,
        "Position": ["Long"] * len(tickers),
        "Entry Price": [np.nan] * len(tickers),
        "Exit Price (optional)": [np.nan] * len(tickers),
        "Shares": [1.0] * len(tickers),
    }
)

edited = st.data_editor(
    base_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Position": st.column_config.SelectboxColumn(
            options=["Long", "Short"],
            help="Long = buy first. Short = sell first (profit if price falls).",
        ),
        "Entry Price": st.column_config.NumberColumn(
            min_value=0.0,
            step=0.01,
            help="Long: buy price. Short: short-sell price.",
        ),
        "Exit Price (optional)": st.column_config.NumberColumn(
            min_value=0.0,
            step=0.01,
            help="Optional. Long: sell price. Short: cover price.",
        ),
        "Shares": st.column_config.NumberColumn(min_value=0.0, step=1.0),
    },
)

run = st.button("Run", type="primary", use_container_width=True)
if not run:
    st.stop()

# ---------------- Validate & normalize inputs ----------------
inputs = edited.copy()
inputs["Ticker"] = inputs["Ticker"].astype(str).str.upper().str.strip()
inputs["Position"] = inputs["Position"].astype(str)

inputs["Shares"] = pd.to_numeric(inputs["Shares"], errors="coerce").fillna(1.0)
inputs.loc[inputs["Shares"] <= 0, "Shares"] = 1.0

inputs["Entry Price"] = pd.to_numeric(inputs["Entry Price"], errors="coerce")
if inputs["Entry Price"].isna().any():
    st.error("Please fill in Entry Price for all tickers.")
    st.stop()

inputs["Exit Price (optional)"] = pd.to_numeric(inputs["Exit Price (optional)"], errors="coerce")

# ---------------- Latest prices + snapshot metrics ----------------
with st.spinner("Fetching latest prices..."):
    latest = get_latest_prices(list(inputs["Ticker"]))

missing = [t for t in inputs["Ticker"] if t not in latest.index]
if missing:
    st.warning(f"Some tickers returned no recent price and will be skipped: {', '.join(missing)}")

inputs = inputs[inputs["Ticker"].isin(latest.index)].copy()
if inputs.empty:
    st.error("No usable tickers after price fetch.")
    st.stop()

inputs["Last Price"] = inputs["Ticker"].map(latest.to_dict()).astype(float)

# If user provided Exit Price, use it as reference; otherwise use latest
inputs["Ref Price"] = np.where(
    inputs["Exit Price (optional)"].notna(),
    inputs["Exit Price (optional)"],
    inputs["Last Price"],
).astype(float)

# Exposure (absolute) used for weights
inputs["Exposure"] = (inputs["Entry Price"] * inputs["Shares"]).abs()

# Signed notional for informational net exposure
inputs["Signed Notional"] = np.where(
    inputs["Position"] == "Long",
    inputs["Entry Price"] * inputs["Shares"],
    -inputs["Entry Price"] * inputs["Shares"],
)

# Signed mark value
inputs["Signed Mkt Value"] = np.where(
    inputs["Position"] == "Long",
    inputs["Last Price"] * inputs["Shares"],
    -inputs["Last Price"] * inputs["Shares"],
)

# P/L in currency
inputs["P/L"] = np.where(
    inputs["Position"] == "Long",
    (inputs["Ref Price"] - inputs["Entry Price"]) * inputs["Shares"],
    (inputs["Entry Price"] - inputs["Ref Price"]) * inputs["Shares"],
)

# P/L % vs exposure
inputs["P/L %"] = np.where(inputs["Exposure"] > 0, inputs["P/L"] / inputs["Exposure"], np.nan)

total_exposure = float(inputs["Exposure"].sum())
net_notional = float(inputs["Signed Notional"].sum())
total_pl = float(inputs["P/L"].sum())

m1, m2, m3 = st.columns(3)
m1.metric("Total Exposure", f"{total_exposure:,.2f}")
m2.metric("Net Notional", f"{net_notional:,.2f}")
m3.metric("Total P/L", f"{total_pl:,.2f}")

inputs["Weight"] = np.where(total_exposure > 0, inputs["Exposure"] / total_exposure, np.nan)

# ---------------- Allocation visuals ----------------
if show_exposure_pie:
    st.markdown("### Allocation (by exposure)")
    fig_pie = px.pie(
        inputs,
        names="Ticker",
        values="Exposure",
        hole=0.45,
        template="plotly_white",
    )
    fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("### Holdings summary")
show_df = inputs[
    ["Ticker", "Position", "Shares", "Entry Price", "Last Price", "Exit Price (optional)", "Exposure", "P/L", "P/L %", "Weight"]
].copy()
show_df["P/L %"] = show_df["P/L %"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
show_df["Weight"] = show_df["Weight"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
st.dataframe(show_df, use_container_width=True)

# ---------------- Portfolio performance (index=100) ----------------
st.markdown("## Portfolio Performance")

with st.spinner("Downloading historical prices..."):
    price_hist = download_adj_close(list(inputs["Ticker"]), start_date, end_date)

if price_hist is None or price_hist.empty:
    st.error("No historical data returned for the selected date range.")
    st.stop()

# Clean and align
price_hist = price_hist.dropna(axis=1, how="all").ffill().dropna()
if price_hist.empty:
    st.error("No historical data after cleaning (all NaNs).")
    st.stop()

hold = inputs.copy()
hold["Ticker"] = hold["Ticker"].astype(str).str.upper().str.strip()
hold = hold.set_index("Ticker")

tickers_used = [t for t in price_hist.columns.astype(str) if t in hold.index]
if not tickers_used:
    st.error("Downloaded history doesn't match your tickers (after cleaning).")
    st.stop()

price_hist = price_hist[tickers_used]
hold = hold.reindex(tickers_used)

prices = price_hist.values.astype(float)                 # (T, N)
shares = hold["Shares"].astype(float).values             # (N,)
entry = hold["Entry Price"].astype(float).values         # (N,)
pos = hold["Position"].astype(str).values                # (N,)

# Per-ticker P/L over time
pl_tn = np.where(
    (pos == "Long")[None, :],
    (prices - entry[None, :]) * shares[None, :],
    (entry[None, :] - prices) * shares[None, :],
)

pl_t = pl_tn.sum(axis=1)

# Start equity = total absolute entry exposure (gives stable index=100)
abs_exposure = float(np.sum(np.abs(entry * shares)))
if abs_exposure <= 0:
    abs_exposure = 1.0

equity = abs_exposure + pl_t
if equity[0] == 0:
    st.error("Equity at start is 0; cannot compute indexed performance.")
    st.stop()

equity_index = 100.0 * (equity / equity[0])

perf_df = pd.DataFrame(
    {"Date": price_hist.index, "Portfolio (start=100)": equity_index, "Equity ($)": equity, "P/L ($)": pl_t}
)

fig_perf = go.Figure()
fig_perf.add_trace(
    go.Scatter(
        x=perf_df["Date"],
        y=perf_df["Portfolio (start=100)"],
        mode="lines",
        name="Portfolio",
        hovertemplate="Date: %{x}<br>Index: %{y:.2f}<extra></extra>",
    )
)
fig_perf.update_layout(
    template="plotly_white",
    height=520,
    margin=dict(l=20, r=20, t=10, b=40),
)
fig_perf.update_xaxes(title="Date", rangeslider=dict(visible=True))
fig_perf.update_yaxes(title="Portfolio (start=100)")
st.plotly_chart(fig_perf, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

# Drawdown
st.markdown("### Drawdown")
equity_series = pd.Series(equity, index=price_hist.index)
drawdown = equity_series / equity_series.cummax() - 1.0
dd_df = pd.DataFrame({"Date": drawdown.index, "Drawdown": drawdown.values})

fig_dd = go.Figure()
fig_dd.add_trace(
    go.Scatter(
        x=dd_df["Date"],
        y=dd_df["Drawdown"],
        mode="lines",
        name="Drawdown",
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>",
    )
)
fig_dd.update_layout(
    template="plotly_white",
    height=260,
    margin=dict(l=20, r=20, t=10, b=40),
)
fig_dd.update_xaxes(title="Date", rangeslider=dict(visible=True))
fig_dd.update_yaxes(title="Drawdown", tickformat=".0%")
st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

# Debug: per-ticker P/L
if show_debug:
    with st.expander("Show per-ticker P/L lines (debug)"):
        per_df = pd.DataFrame(pl_tn, index=price_hist.index, columns=tickers_used)
        st.dataframe(per_df, use_container_width=True)
