# pages/4_MA_Signal_Backtest.py

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
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
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Try another symbol/date range.")

    # Ensure we have Close
    if "Close" not in df.columns:
        raise ValueError(f"Ticker '{ticker}' returned no 'Close' column. Columns: {list(df.columns)}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def build_signal(df: pd.DataFrame, buy_rule: str) -> pd.DataFrame:
    out = df.copy()

    # buy_rule determines which crossover means +1
    if buy_rule == "Buy when MA60 > MA15":
        buy = out["MA60"] > out["MA15"]
        sell = out["MA60"] < out["MA15"]
    else:
        buy = out["MA15"] > out["MA60"]
        sell = out["MA15"] < out["MA60"]

    out["Signal"] = 0
    out.loc[buy, "Signal"] = 1
    out.loc[sell, "Signal"] = -1

    # Only allow signals after both MAs exist
    valid = out["MA15"].notna() & out["MA60"].notna()
    out.loc[~valid, "Signal"] = 0
    return out


def add_buy_regime_vrects(fig: go.Figure, dates: pd.Series, signal: pd.Series):
    """Shade +1 regimes as vertical rectangles."""
    s = signal.fillna(0).astype(int).values
    d = pd.to_datetime(dates).values

    in_regime = False
    start = None

    for i in range(len(s)):
        is_on = (s[i] == 1)

        if (not in_regime) and is_on:
            in_regime = True
            start = d[i]

        if in_regime and ((not is_on) or i == len(s) - 1):
            end = d[i]
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="green",
                opacity=0.10,
                line_width=0,
                layer="below",
            )
            in_regime = False
            start = None


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="MA Signal Backtest", layout="wide")

st.title("MA Signal Backtest")
st.caption("MA15/MA60 regime signal + next-day-return backtest.")

with st.sidebar:
    st.header("Inputs")

    ticker = normalize_ticker(st.text_input("Ticker", value="AAPL"))

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 3)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=today)

    ma_short = st.number_input("Short MA window", min_value=2, max_value=400, value=15, step=1)
    ma_long = st.number_input("Long MA window", min_value=3, max_value=800, value=60, step=1)

    st.divider()
    buy_rule = st.radio(
        "Buy / Sell rule",
        options=["Buy when MA60 > MA15", "Buy when MA15 > MA60"],
        index=0,
    )

    run = st.button("Run", type="primary", use_container_width=True)

if not run:
    st.info("Choose inputs on the left and click **Run**.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if ma_long <= ma_short:
    st.warning("Usually long MA should be > short MA. (Not required, but results may be strange.)")


# -----------------------------
# Download + clean
# -----------------------------
try:
    with st.spinner("Downloading price data..."):
        raw = download_prices(ticker, start_date, end_date)
except Exception as e:
    st.error(str(e))
    st.stop()

df = raw[["Close"]].copy()

# Force numeric Close (this fixes “blank chart” cases)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

if df.empty:
    st.error("After cleaning, Close prices are empty (all NaN). Try a different ticker/date range.")
    st.stop()

# Build MAs
df["MA15"] = df["Close"].rolling(int(ma_short)).mean()
df["MA60"] = df["Close"].rolling(int(ma_long)).mean()

# Signal
df = build_signal(df, buy_rule)

# Backtest: next-day returns
df["returns"] = df["Close"].pct_change().shift(-1)
df["strategy_returns"] = df["Signal"] * df["returns"]
df["cumulative_strategy_returns"] = (1 + df["strategy_returns"].fillna(0)).cumprod()

# Reset index into a REAL Date column (Plotly reliability)
df = df.reset_index().rename(columns={"index": "Date"})
df["Date"] = pd.to_datetime(df["Date"])


# -----------------------------
# Metrics
# -----------------------------
total_growth = float(df["cumulative_strategy_returns"].iloc[-1])
total_return = (total_growth - 1) * 100

pct_long = (df["Signal"] == 1).mean() * 100
pct_short = (df["Signal"] == -1).mean() * 100

c1, c2, c3 = st.columns(3)
c1.metric("Cumulative growth", f"{total_growth:.3f}x")
c2.metric("Total strategy return", f"{total_return:.2f}%")
c3.metric("Time in +1 / -1", f"{pct_long:.1f}% / {pct_short:.1f}%")


# -----------------------------
# Chart 1: Price + MAs + Regime
# -----------------------------
st.subheader("Price + MAs + Regime")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
fig1.add_trace(go.Scatter(x=df["Date"], y=df["MA15"], mode="lines", name=f"MA{ma_short}"))
fig1.add_trace(go.Scatter(x=df["Date"], y=df["MA60"], mode="lines", name=f"MA{ma_long}", line=dict(dash="dash")))

add_buy_regime_vrects(fig1, df["Date"], df["Signal"])

fig1.update_layout(
    title=f"{ticker} — Close with MAs + Regime ({buy_rule})",
    xaxis_title="Date",
    yaxis_title="Price",
    height=520,
    margin=dict(l=10, r=10, t=60, b=10),
    template="plotly_dark",  # ensures visibility on dark theme
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig1, use_container_width=True)


# -----------------------------
# Chart 2: Signal
# -----------------------------
st.subheader("Signal (+1 / -1)")
fig2 = px.line(df, x="Date", y="Signal", title=f"{ticker} — Signal ({buy_rule})")
fig2.update_traces(line_shape="hv")
fig2.update_layout(height=320, template="plotly_dark", margin=dict(l=10, r=10, t=60, b=10))
fig2.update_yaxes(range=[-1.1, 1.1])
st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# Chart 3: Cumulative strategy
# -----------------------------
st.subheader("Cumulative Strategy Returns")
fig3 = px.line(df, x="Date", y="cumulative_strategy_returns", title=f"{ticker} — Growth of $1")
fig3.update_layout(height=360, template="plotly_dark", margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig3, use_container_width=True)


with st.expander("Debug: show cleaned data"):
    st.write("If charts are blank, check Close/Date values here:")
    st.dataframe(df[["Date", "Close", "MA15", "MA60", "Signal"]].tail(50), use_container_width=True)
