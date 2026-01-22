import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="Portfolio from Buy Prices", layout="wide")
st.title("Portfolio Allocation + Performance (from Buy Price)")
st.caption("Enter tickers + buy price (and shares). App pulls prices from Yahoo Finance to compute allocation and performance.")

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
        auto_adjust=True,     # adjusted close becomes "Close"
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
    return close

@st.cache_data(show_spinner=False)
def get_latest_prices(tickers: list[str]) -> pd.Series:
    # Uses last available close from recent history (reliable vs fast_info blocks)
    end = dt.date.today()
    start = end - dt.timedelta(days=10)
    px_df = download_adj_close(tickers, start, end)
    if px_df.empty:
        return pd.Series(dtype=float)
    last = px_df.dropna().iloc[-1]
    last.index = last.index.astype(str)
    return last.astype(float)

# ---------------- UI ----------------
tickers_raw = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, NVDA", height=80)
tickers = normalize_tickers(tickers_raw)

today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 2)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Performance start date", value=default_start)
with c2:
    end_date = st.date_input("Performance end date", value=today)

st.markdown("### Buy inputs")
st.write("Enter **Buy Price** and **Shares**. If Shares is blank/0, it will default to 1.")

if not tickers:
    st.info("Add at least one ticker.")
    st.stop()

# Editable table
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

# Validate inputs
inputs = edited.copy()
inputs["Ticker"] = inputs["Ticker"].astype(str).str.upper()

# Shares default to 1 if missing/0
inputs["Shares"] = pd.to_numeric(inputs["Shares"], errors="coerce").fillna(1.0)
inputs.loc[inputs["Shares"] <= 0, "Shares"] = 1.0

# Buy Price required
inputs["Buy Price"] = pd.to_numeric(inputs["Buy Price"], errors="coerce")
if inputs["Buy Price"].isna().any():
    st.error("Please fill in Buy Price for all tickers.")
    st.stop()

# ---------------- Current allocation ----------------
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
inputs["Cost Basis"] = inputs["Buy Price"] * inputs["Shares"]
inputs["Current Value"] = inputs["Last Price"] * inputs["Shares"]
inputs["P/L"] = inputs["Current Value"] - inputs["Cost Basis"]
inputs["P/L %"] = np.where(inputs["Cost Basis"] > 0, inputs["P/L"] / inputs["Cost Basis"], np.nan)

total_value = float(inputs["Current Value"].sum())
inputs["Weight"] = inputs["Current Value"] / total_value

m1, m2, m3 = st.columns(3)
m1.metric("Portfolio Value", f"{total_value:,.2f}")
m2.metric("Cost Basis", f"{float(inputs['Cost Basis'].sum()):,.2f}")
m3.metric("Total P/L", f"{float(inputs['P/L'].sum()):,.2f}")

st.markdown("### Allocation (by current market value)")
fig_pie = px.pie(
    inputs,
    names="Ticker",
    values="Current Value",
    hole=0.45,
    template="plotly_white",
)
fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("### Holdings summary")
show_df = inputs[["Ticker", "Shares", "Buy Price", "Last Price", "Cost Basis", "Current Value", "P/L", "P/L %", "Weight"]].copy()
show_df["P/L %"] = show_df["P/L %"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
show_df["Weight"] = show_df["Weight"].map(lambda x: f"{x:.2%}")
st.dataframe(show_df, use_container_width=True)

# ---------------- Performance chart ----------------
st.markdown("### Portfolio performance")

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner("Downloading historical prices for performance..."):
    price_hist = download_adj_close(list(inputs["Ticker"]), start_date, end_date)

price_hist = price_hist.dropna(axis=1, how="all").ffill().dropna()
if price_hist.empty:
    st.error("No historical data for the selected date range.")
    st.stop()

# Align holdings to columns returned
holdings = inputs.set_index("Ticker").reindex(price_hist.columns)

shares = holdings["Shares"].astype(float).values
# Portfolio value over time = sum(price * shares)
port_value = (price_hist.values * shares).sum(axis=1)
port_value = pd.Series(port_value, index=price_hist.index, name="Portfolio Value")

# Normalize to 100 for a clean performance plot
port_norm = 100 * (port_value / port_value.iloc[0])
perf_df = pd.DataFrame({"Date": port_norm.index, "Portfolio (start=100)": port_norm.values})

fig_perf = px.line(perf_df, x="Date", y="Portfolio (start=100)", template="plotly_white")
fig_perf.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=40))
st.plotly_chart(fig_perf, use_container_width=True)

# Optional: show drawdown
dd = port_value / port_value.cummax() - 1
dd_df = pd.DataFrame({"Date": dd.index, "Drawdown": dd.values})
fig_dd = px.area(dd_df, x="Date", y="Drawdown", template="plotly_white")
fig_dd.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=40))
fig_dd.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_dd, use_container_width=True)
