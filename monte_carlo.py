# app.py — Streamlit Portfolio Allocation + Dollar Performance (Yahoo tickers)
# Fixes common AAPL/NVDA issues by:
# - robust ticker parsing
# - robust yfinance multi/single ticker handling
# - safe forward-fill + alignment

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="Portfolio Visualizer", layout="wide")
st.title("Portfolio Visualizer (Buy Price + Shares)")
st.caption("Enter tickers, buy price, and shares. Shows allocation (current value) and portfolio performance in dollars.")

# ----------------------------
# Helpers
# ----------------------------
def normalize_tickers(raw: str) -> list[str]:
    raw = raw.replace("\n", ",").replace(" ", ",")
    parts = [t.strip().upper().replace("ˆ", "^") for t in raw.split(",") if t.strip()]

    # unique keep order
    seen, out = set(), []
    for t in parts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


@st.cache_data(show_spinner=False)
def download_close(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Downloads adjusted close as 'Close' using auto_adjust=True.
    Returns DataFrame columns=tickers.
    Handles both single and multi-ticker yfinance output reliably.
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

    # Multi-ticker: MultiIndex columns like ('Close','AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()
        close = df["Close"].copy()
        # Normalize column names
        close.columns = [str(c).upper() for c in close.columns]
    else:
        # Single ticker: normal columns include 'Close'
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df[["Close"]].copy()
        close.columns = [tickers[0].upper()]

    # Clean and fill
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close.dropna(how="all").ffill()

    return close


@st.cache_data(show_spinner=False)
def get_latest_prices(tickers: list[str]) -> pd.Series:
    """
    Gets last available adjusted close for each ticker from last ~15 days.
    (More reliable than fast_info which can be blocked.)
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=15)
    px_df = download_close(tickers, start, end)
    if px_df.empty:
        return pd.Series(dtype=float)
    last = px_df.dropna().iloc[-1].astype(float)
    last.index = last.index.astype(str).str.upper()
    return last


def money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


# ----------------------------
# Inputs
# ----------------------------
tickers_raw = st.text_area("Tickers (comma-separated)", value="AAPL, NVDA, MSFT", height=70)
tickers = normalize_tickers(tickers_raw)

today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 2)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Performance start date", value=default_start)
with c2:
    end_date = st.date_input("Performance end date", value=today)

st.markdown("### Holdings")
st.write("Enter **Buy Price** and **Shares** for each ticker. Shares default to 1 if empty/0.")

if not tickers:
    st.info("Please enter at least one ticker.")
    st.stop()

base_df = pd.DataFrame(
    {
        "Ticker": tickers,
        "Buy Price": [np.nan] * len(tickers),
        "Shares": [1.0] * len(tickers),
    }
)

edited = st.data_editor(
    base_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Buy Price": st.column_config.NumberColumn(min_value=0.0, step=0.01),
        "Shares": st.column_config.NumberColumn(min_value=0.0, step=1.0),
    },
)

run = st.button("Run", type="primary", use_container_width=True)
if not run:
    st.stop()

# ----------------------------
# Validate holdings inputs
# ----------------------------
inputs = edited.copy()
inputs["Ticker"] = inputs["Ticker"].astype(str).str.upper()

inputs["Shares"] = pd.to_numeric(inputs["Shares"], errors="coerce").fillna(1.0)
inputs.loc[inputs["Shares"] <= 0, "Shares"] = 1.0

inputs["Buy Price"] = pd.to_numeric(inputs["Buy Price"], errors="coerce")
if inputs["Buy Price"].isna().any():
    st.error("Please fill **Buy Price** for all tickers.")
    st.stop()

# ----------------------------
# Latest prices + allocation
# ----------------------------
with st.spinner("Fetching latest prices..."):
    latest = get_latest_prices(inputs["Ticker"].tolist())

missing = [t for t in inputs["Ticker"] if t not in latest.index]
if missing:
    st.warning(f"These tickers returned no recent price and were skipped: {', '.join(missing)}")

inputs = inputs[inputs["Ticker"].isin(latest.index)].copy()
if inputs.empty:
    st.error("No usable tickers after price fetch. Check tickers.")
    st.stop()

inputs["Last Price"] = inputs["Ticker"].map(latest.to_dict()).astype(float)
inputs["Cost Basis"] = inputs["Buy Price"] * inputs["Shares"]
inputs["Current Value"] = inputs["Last Price"] * inputs["Shares"]
inputs["P/L"] = inputs["Current Value"] - inputs["Cost Basis"]
inputs["P/L %"] = np.where(inputs["Cost Basis"] > 0, inputs["P/L"] / inputs["Cost Basis"], np.nan)

total_value = float(inputs["Current Value"].sum())
inputs["Weight"] = inputs["Current Value"] / total_value

m1, m2, m3, m4 = st.columns(4)
m1.metric("Portfolio Value", money(total_value))
m2.metric("Cost Basis", money(float(inputs["Cost Basis"].sum())))
m3.metric("Total P/L", money(float(inputs["P/L"].sum())))
m4.metric("Total Return", f"{(float(inputs['P/L'].sum()) / float(inputs['Cost Basis'].sum()))*100:.2f}%")

st.markdown("### Allocation (by current market value)")
fig_pie = px.pie(inputs, names="Ticker", values="Current Value", hole=0.45, template="plotly_white")
fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("### Holdings table")
show_df = inputs[["Ticker", "Shares", "Buy Price", "Last Price", "Cost Basis", "Current Value", "P/L", "P/L %", "Weight"]].copy()
show_df["P/L %"] = show_df["P/L %"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
show_df["Weight"] = show_df["Weight"].map(lambda x: f"{x:.2%}")
st.dataframe(show_df, use_container_width=True)

# ----------------------------
# Performance in dollars (not normalized)
# ----------------------------
st.markdown("### Portfolio performance (dollars)")

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner("Downloading historical prices..."):
    price_hist = download_close(inputs["Ticker"].tolist(), start_date, end_date)

price_hist = price_hist.dropna(axis=1, how="all").ffill()

# align holdings to downloaded columns
price_hist = price_hist.dropna(how="any")  # common dates across assets
if price_hist.empty:
    st.error("No historical data after aligning dates. Try a shorter date range or fewer tickers.")
    st.stop()

holdings = inputs.set_index("Ticker").reindex(price_hist.columns)
shares = holdings["Shares"].astype(float).values

# Dollar value through time
port_value = (price_hist.values * shares).sum(axis=1)
port_value = pd.Series(port_value, index=price_hist.index, name="Portfolio Value ($)")

perf_df = pd.DataFrame({"Date": port_value.index, "Portfolio Value ($)": port_value.values})

fig_perf = px.line(perf_df, x="Date", y="Portfolio Value ($)", template="plotly_white")
fig_perf.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=40))
fig_perf.update_xaxes(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
fig_perf.update_yaxes(title="Portfolio Value ($)", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
st.plotly_chart(fig_perf, use_container_width=True)

# Optional: drawdown
dd = port_value / port_value.cummax() - 1
dd_df = pd.DataFrame({"Date": dd.index, "Drawdown": dd.values})
fig_dd = px.area(dd_df, x="Date", y="Drawdown", template="plotly_white")
fig_dd.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=40))
fig_dd.update_xaxes(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
fig_dd.update_yaxes(title="Drawdown", tickformat=".0%", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
st.plotly_chart(fig_dd, use_container_width=True)

# Debug info (helpful if tickers "fail")
with st.expander("Debug (ticker/data checks)"):
    st.write("Parsed tickers:", tickers)
    st.write("Latest prices used:", latest.to_dict())
    st.write("Historical columns returned:", list(price_hist.columns))
    st.write("Historical date range:", price_hist.index.min(), "→", price_hist.index.max())
