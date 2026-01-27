# pages/3_Compare_Two_Tickers.py
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# Helpers
# -----------------------------
def normalize_ticker(t: str) -> str:
    # fixes mac "ˆ" vs "^", trims spaces, uppercases
    return (t or "").strip().upper().replace("ˆ", "^")


@st.cache_data(show_spinner=False)
def download_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    ticker = normalize_ticker(ticker)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range.")

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if col not in df.columns:
        raise ValueError(f"{ticker}: missing Close/Adj Close. Columns: {list(df.columns)}")

    s = df[col].dropna().astype(float)
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s


def to_log_returns(px: pd.Series) -> pd.Series:
    r = np.log(px / px.shift(1)).dropna()
    r.name = f"{px.name}_logret"
    return r


def plot_two_series(
    x1: pd.Series,
    x2: pd.Series,
    title: str,
    ytitle: str,
    dashed_second: bool = False,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x1.index, y=x1.values, mode="lines", name=x1.name
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x2.index,
            y=x2.values,
            mode="lines",
            name=x2.name,
            line=dict(dash="dash" if dashed_second else "solid"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=ytitle,
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def plot_one_series(x: pd.Series, title: str, ytitle: str, add_zero_line: bool = False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.index, y=x.values, mode="lines", name=x.name))
    if add_zero_line:
        fig.add_hline(y=0)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=ytitle,
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Stock Correlation Simulation")
st.caption("View how stocks correlate to one another.")

with st.sidebar:
    st.header("Inputs")

    c1, c2 = st.columns(2)
    with c1:
        t1 = st.text_input("Ticker 1")
    with c2:
        t2 = st.text_input("Ticker 2")

    t1 = normalize_ticker(t1)
    t2 = normalize_ticker(t2)

    years = st.slider("Years of data", min_value=1, max_value=20, value=3, step=1)
    window = st.slider("Rolling correlation window (trading days)", min_value=5, max_value=252, value=21, step=1)

    st.divider()
    run = st.button("Run", type="primary", use_container_width=True)

if not run:
    st.info("Set tickers + years on the left, then click **Run**.")
    st.stop()

if not t1 or not t2:
    st.error("Please enter both tickers.")
    st.stop()

if t1 == t2:
    st.warning("You chose the same ticker twice. Change one to compare.")
    st.stop()

end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
start = end - pd.Timedelta(days=int(365 * years))

# -----------------------------
# Data
# -----------------------------
try:
    with st.spinner("Downloading prices..."):
        s1 = download_close(t1, start=start, end=end)
        s2 = download_close(t2, start=start, end=end)
except Exception as e:
    st.error(f"Download error: {e}")
    st.stop()

# Align dates
data = pd.concat([s1, s2], axis=1).dropna()
if data.empty:
    st.error("No overlapping dates between the two series in this range. Try a longer range.")
    st.stop()

s1 = data[t1]
s2 = data[t2]

# Returns
r1 = to_log_returns(s1)
r2 = to_log_returns(s2)

# Normalized prices
s1_norm = (100 * (s1 / s1.iloc[0])).rename(t1)
s2_norm = (100 * (s2 / s2.iloc[0])).rename(t2)

# Cumulative returns (from log returns)
cum1 = (np.exp(r1.cumsum()) - 1.0).rename(t1)
cum2 = (np.exp(r2.cumsum()) - 1.0).rename(t2)

# Rolling correlation
rolling_corr = r1.rolling(int(window)).corr(r2)
rolling_corr.name = f"{window}D rolling corr"

# -----------------------------
# Quick stats (top cards)
# -----------------------------
corr = float(r1.corr(r2))
t1_total = float(cum1.iloc[-1] * 100)
t2_total = float(cum2.iloc[-1] * 100)

m1, m2, m3 = st.columns(3)
m1.metric("Return correlation", f"{corr:.3f}")
m2.metric(f"{t1} total return", f"{t1_total:.2f}%")
m3.metric(f"{t2} total return", f"{t2_total:.2f}%")

# -----------------------------
# Charts (Streamlit-like style: clean, stacked, interactive)
# -----------------------------
st.subheader("Normalized Prices (start = 100)")
fig_norm = plot_two_series(
    s1_norm.rename(t1),
    s2_norm.rename(t2),
    title=f"{t1} vs {t2} — Normalized Prices",
    ytitle="Index (start = 100)",
    dashed_second=True,
)
st.plotly_chart(fig_norm, use_container_width=True)

st.subheader("Daily Log Returns")
rets_pct = pd.concat([(r1 * 100).rename(t1), (r2 * 100).rename(t2)], axis=1).dropna()
fig_rets = plot_two_series(
    rets_pct[t1].rename(t1),
    rets_pct[t2].rename(t2),
    title=f"{t1} vs {t2} — Daily Log Returns",
    ytitle="Return (%)",
    dashed_second=True,
)
st.plotly_chart(fig_rets, use_container_width=True)

st.subheader(f"Rolling Correlation ({window} days)")
fig_corr = plot_one_series(
    rolling_corr.dropna(),
    title=f"{t1} vs {t2} — Rolling Correlation",
    ytitle="Correlation",
    add_zero_line=True,
)
st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------
# Data table (optional)
# -----------------------------
with st.expander("Show data"):
    out = pd.DataFrame(
        {
            f"{t1} Close": s1,
            f"{t2} Close": s2,
            f"{t1} Norm": s1_norm,
            f"{t2} Norm": s2_norm,
            f"{t1} LogRet": r1,
            f"{t2} LogRet": r2,
            "Rolling Corr": rolling_corr,
        }
    ).dropna(how="all")
    st.dataframe(out.tail(250), use_container_width=True)
