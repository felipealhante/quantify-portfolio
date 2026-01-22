# pages/4_MA_Signal_Backtest.py

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# Helpers
# -----------------------------
def normalize_ticker(t: str) -> str:
    return (t or "").strip().upper().replace("ˆ", "^")


@st.cache_data(show_spinner=False)
def download_prices(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"No valid data for ticker '{ticker}'. Try another symbol or date range.")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def make_price_ma_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="Close"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA15"], mode="lines", name="MA15"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA60"], mode="lines", name="MA60",
        line=dict(dash="dash")
    ))

    # Shade regime where Signal == +1
    bull = (df["Signal"] == 1).fillna(False).astype(bool)
    # Create a filled area by plotting an "invisible" series
    # We fill between min/max of close for the selected bull dates.
    if bull.any():
        y0 = float(df["Close"].min())
        y1 = float(df["Close"].max())
        fig.add_trace(go.Scatter(
            x=df.index[bull],
            y=[y1] * bull.sum(),
            mode="lines",
            line=dict(width=0),
            showlegend=True,
            name="Signal = +1 (MA60 > MA15)",
            fill="tonexty",
        ))
        fig.add_trace(go.Scatter(
            x=df.index[bull],
            y=[y0] * bull.sum(),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        ))
        # Reorder fill traces behind other lines
        fig.data = (fig.data[:3] + (fig.data[4], fig.data[3]))

    fig.update_layout(
        title=f"{ticker} — Close with MA15/MA60 + Signal Regime",
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def make_signal_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Signal"],
        mode="lines",
        name="Signal",
        line_shape="hv"  # step-like
    ))
    fig.add_hline(y=0)
    fig.update_layout(
        title=f"{ticker} — Signal (1 when MA60>MA15, -1 when MA60<MA15)",
        xaxis_title="Date",
        yaxis_title="Signal",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(range=[-1.1, 1.1], showgrid=True)
    fig.update_xaxes(showgrid=True)
    return fig


def make_cum_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["cumulative_strategy_returns"],
        mode="lines",
        name="Cumulative strategy",
    ))
    fig.update_layout(
        title=f"{ticker} — Cumulative Strategy Returns",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("MA Signal Backtest")
st.caption("MA15 / MA60 regime signal + next-day-return backtest (same logic as your script).")

with st.sidebar:
    st.header("Inputs")

    ticker = normalize_ticker(st.text_input("Ticker", value="AAPL"))

    # Default dates
    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 3)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=today)

    ma_short = st.number_input("Short MA window (MA15)", min_value=2, max_value=400, value=15, step=1)
    ma_long = st.number_input("Long MA window (MA60)", min_value=5, max_value=800, value=60, step=1)

    st.divider()
    run = st.button("Run", type="primary", use_container_width=True)

if not run:
    st.info("Set your inputs on the left, then click **Run**.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if ma_long <= ma_short:
    st.warning("Usually MA60 > MA15. Your long MA is not larger than the short MA.")

# -----------------------------
# Data + calculations
# -----------------------------
try:
    with st.spinner("Downloading data..."):
        data = download_prices(ticker, start_date, end_date)
except Exception as e:
    st.error(str(e))
    st.stop()

df = data[["Close"]].dropna().copy()
df["MA15"] = df["Close"].rolling(window=int(ma_short)).mean()
df["MA60"] = df["Close"].rolling(window=int(ma_long)).mean()

# Your naming (slow=MA15, fast=MA60) + your logic
slow_ma = df["MA15"]
fast_ma = df["MA60"]

buy_signal = (fast_ma > slow_ma)
sell_signal = (fast_ma < slow_ma)

df["Signal"] = 0
df.loc[buy_signal, "Signal"] = 1
df.loc[sell_signal, "Signal"] = -1

# keep 0 until both MAs exist
valid = df["MA15"].notna() & df["MA60"].notna()
df.loc[~valid, "Signal"] = 0

# Backtest next-day returns
df["returns"] = df["Close"].pct_change().shift(-1)
df["strategy_returns"] = df["Signal"] * df["returns"]
df["cumulative_strategy_returns"] = (1 + df["strategy_returns"].fillna(0)).cumprod()

# -----------------------------
# Metrics
# -----------------------------
total_growth = float(df["cumulative_strategy_returns"].iloc[-1])
total_return = (total_growth - 1.0) * 100

signal_counts = df["Signal"].value_counts(dropna=True).to_dict()
pct_long = 100.0 * signal_counts.get(1, 0) / max(1, len(df))
pct_short = 100.0 * signal_counts.get(-1, 0) / max(1, len(df))

c1, c2, c3 = st.columns(3)
c1.metric("Cumulative growth", f"{total_growth:.3f}x")
c2.metric("Total strategy return", f"{total_return:.2f}%")
c3.metric("Time in +1 / -1", f"{pct_long:.1f}% / {pct_short:.1f}%")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Price + MAs + Regime")
st.plotly_chart(make_price_ma_figure(df, ticker), use_container_width=True)

st.subheader("Signal")
st.plotly_chart(make_signal_figure(df, ticker), use_container_width=True)

st.subheader("Cumulative Strategy Returns")
st.plotly_chart(make_cum_figure(df, ticker), use_container_width=True)

# -----------------------------
# Table
# -----------------------------
with st.expander("Last rows"):
    cols = ["Close", "MA15", "MA60", "Signal", "returns", "strategy_returns", "cumulative_strategy_returns"]
    st.dataframe(df[cols].tail(25), use_container_width=True)
