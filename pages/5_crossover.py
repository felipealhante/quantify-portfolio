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


def build_signal(df: pd.DataFrame, buy_rule: str) -> pd.DataFrame:
    """
    buy_rule:
      - "MA60 > MA15 (Buy)"  => Signal=+1 when MA60>MA15, -1 when MA60<MA15
      - "MA15 > MA60 (Buy)"  => Signal=+1 when MA15>MA60, -1 when MA15<MA60
    """
    out = df.copy()

    if buy_rule.startswith("MA60"):
        buy = out["MA60"] > out["MA15"]
        sell = out["MA60"] < out["MA15"]
    else:
        buy = out["MA15"] > out["MA60"]
        sell = out["MA15"] < out["MA60"]

    out["Signal"] = 0
    out.loc[buy, "Signal"] = 1
    out.loc[sell, "Signal"] = -1

    # keep 0 until both MAs exist
    valid = out["MA15"].notna() & out["MA60"].notna()
    out.loc[~valid, "Signal"] = 0
    return out


def add_regime_vrects(fig: go.Figure, idx: pd.DatetimeIndex, signal: pd.Series):
    """Shade +1 regimes using vertical rectangles (robust)."""
    s = signal.fillna(0).astype(int)

    in_regime = False
    start = None

    for i in range(len(s)):
        is_on = (s.iloc[i] == 1)

        if (not in_regime) and is_on:
            in_regime = True
            start = idx[i]

        # regime ends when it stops being 1 OR last point
        if in_regime and ((not is_on) or i == len(s) - 1):
            end = idx[i] if not is_on else idx[i]
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="green",
                opacity=0.08,
                line_width=0,
                layer="below",
            )
            in_regime = False
            start = None


def make_price_ma_figure(df: pd.DataFrame, ticker: str, buy_rule: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="Close"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA15"], mode="lines", name="MA15"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA60"], mode="lines", name="MA60", line=dict(dash="dash")
    ))

    # Shade buy regimes (+1)
    add_regime_vrects(fig, df.index, df["Signal"])

    fig.update_layout(
        title=f"{ticker} — Close with MA15/MA60 + Regime ({buy_rule})",
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def make_signal_figure(df: pd.DataFrame, ticker: str, buy_rule: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Signal"],
        mode="lines",
        name="Signal",
        line_shape="hv"
    ))
    fig.add_hline(y=0)
    fig.update_layout(
        title=f"{ticker} — Signal (+1 is Buy Regime) ({buy_rule})",
        xaxis_title="Date",
        yaxis_title="Signal",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
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
        showlegend=False
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("MA Signal Backtest")
st.caption("MA15 / MA60 regime signal + next-day-return backtest.")

with st.sidebar:
    st.header("Inputs")

    ticker = normalize_ticker(st.text_input("Ticker", value="AAPL"))

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 3)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=today)

    ma_short = st.number_input("Short MA window (MA15)", min_value=2, max_value=400, value=15, step=1)
    ma_long = st.number_input("Long MA window (MA60)", min_value=5, max_value=800, value=60, step=1)

    st.divider()
    st.subheader("Signal rule")

    buy_rule = st.radio(
        "Choose which crossover means BUY (+1):",
        options=["MA60 > MA15 (Buy)", "MA15 > MA60 (Buy)"],
        index=0,
    )

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

df = build_signal(df, buy_rule=buy_rule)

# next-day returns backtest
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
# Charts (Plotly, Streamlit style)
# -----------------------------
st.subheader("Price + MAs + Regime")
st.plotly_chart(make_price_ma_figure(df, ticker, buy_rule), use_container_width=True)

st.subheader("Signal")
st.plotly_chart(make_signal_figure(df, ticker, buy_rule), use_container_width=True)

st.subheader("Cumulative Strategy Returns")
st.plotly_chart(make_cum_figure(df, ticker), use_container_width=True)

with st.expander("Last rows"):
    cols = ["Close", "MA15", "MA60", "Signal", "returns", "strategy_returns", "cumulative_strategy_returns"]
    st.dataframe(df[cols].tail(25), use_container_width=True)
