# portfolio_app.py
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Visualizer", layout="wide")

st.markdown(
    "<h1 style='text-align:center; font-size:58px; margin-bottom:6px;'>Portfolio Visualizer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; opacity:0.75; margin-top:0;'>Build, visualize, and analyze a portfolio from Yahoo Finance data.</p>",
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
def normalize_tickers(raw: str) -> list[str]:
    # Split by comma/space/newline, uppercase, remove empties
    parts = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        t = chunk.strip().upper()
        if t:
            parts.append(t)
    # Unique but keep order
    seen = set()
    out = []
    for t in parts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def parse_weights(raw: str, n: int) -> np.ndarray | None:
    """
    raw example: "0.5,0.3,0.2"
    returns normalized weights sum=1, or None if invalid/empty
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        vals = []
        for x in raw.replace("\n", ",").replace(" ", ",").split(","):
            x = x.strip()
            if x:
                vals.append(float(x))
        if len(vals) != n:
            return None
        w = np.array(vals, dtype=float)
        if np.any(w < 0):
            return None
        s = float(w.sum())
        if s <= 0:
            return None
        return w / s
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def download_adj_close(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df.empty:
        return pd.DataFrame()
    # When multiple tickers, yfinance returns columns like ('Close', 'AAPL') or 'Close' etc.
    # With auto_adjust=True, we can use 'Close' as adjusted.
    if isinstance(df.columns, pd.MultiIndex):
        # Grab Close level
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
        else:
            # fallback: take last level if needed
            close = df.xs(df.columns.levels[0][0], axis=1, level=0)
    else:
        # single ticker -> series-like columns
        if "Close" in df.columns:
            close = df[["Close"]].copy()
            close.columns = tickers  # rename to ticker
        else:
            return pd.DataFrame()

    # Ensure all requested tickers present
    close = close.dropna(how="all")
    close = close.fillna(method="ffill").dropna(how="any")
    # If single ticker returned as Series, ensure DataFrame
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close


def portfolio_metrics(port_ret: pd.Series, rf: float = 0.0) -> dict:
    """
    port_ret: daily returns series
    rf: annual risk-free rate (e.g. 0.03)
    """
    if port_ret.empty:
        return {}
    ann_factor = 252
    avg = port_ret.mean() * ann_factor
    vol = port_ret.std(ddof=0) * np.sqrt(ann_factor)

    # Sharpe using annual rf -> convert rf to daily approx
    sharpe = np.nan
    if vol > 0:
        sharpe = (avg - rf) / vol

    # Equity curve + drawdown
    equity = (1 + port_ret).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = dd.min()

    # CAGR
    days = (port_ret.index[-1] - port_ret.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = np.nan
    if years and years > 0:
        cagr = equity.iloc[-1] ** (1 / years) - 1

    return {
        "CAGR": cagr,
        "Annual Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Annual Return": avg,
    }


# ---------------- Sidebar Inputs ----------------
colL, colC, colR = st.columns([1, 2, 1])

with colC:
    tickers_raw = st.text_area(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA, SPY",
        height=70,
    )
    tickers = normalize_tickers(tickers_raw)

    today = dt.date.today()
    start_default = today - dt.timedelta(days=365 * 3)

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=start_default)
    with c2:
        end_date = st.date_input("End date", value=today)

    st.markdown("---")

    weights_raw = st.text_input(
        "Weights (optional, must match tickers count; example: 0.4,0.3,0.3)",
        value="",
    )

    bmk = st.text_input("Benchmark ticker (optional)", value="SPY").strip().upper()
    rf = st.number_input("Risk-free rate (annual, e.g. 0.03 = 3%)", value=0.03, step=0.01, format="%.2f")

    run = st.button("Run", type="primary", use_container_width=True)

if not run:
    st.stop()

if len(tickers) < 1:
    st.error("Add at least one ticker.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

weights = parse_weights(weights_raw, len(tickers))

# ---------------- Download Data ----------------
with st.spinner("Downloading prices..."):
    prices = download_adj_close(tickers, start_date, end_date)

if prices.empty:
    st.error("No data returned. Check tickers/date range.")
    st.stop()

# If some tickers got dropped due to missing data
tickers_final = list(prices.columns)
if tickers_final != tickers:
    st.warning(f"Using tickers with full data: {', '.join(tickers_final)}")
    tickers = tickers_final
    weights = parse_weights(weights_raw, len(tickers))  # re-validate

# Default weights if none given
if weights is None:
    weights = np.ones(len(tickers)) / len(tickers)

# ---------------- Returns & Portfolio ----------------
rets = prices.pct_change().dropna()
port_ret = (rets * weights).sum(axis=1)
equity = (1 + port_ret).cumprod()

# Benchmark
bmk_equity = None
bmk_ret = None
if bmk:
    bmk_prices = download_adj_close([bmk], start_date, end_date)
    if not bmk_prices.empty:
        bmk_ret = bmk_prices.iloc[:, 0].pct_change().dropna()
        # Align dates with portfolio
        bmk_ret = bmk_ret.reindex(port_ret.index).dropna()
        if len(bmk_ret) > 5:
            bmk_equity = (1 + bmk_ret).cumprod()

# ---------------- Metrics ----------------
m = portfolio_metrics(port_ret, rf=rf)
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Tickers", f"{len(tickers)}")
col2.metric("CAGR", f"{m['CAGR']*100:,.2f}%" if pd.notna(m["CAGR"]) else "—")
col3.metric("Annual Vol", f"{m['Annual Vol']*100:,.2f}%" if pd.notna(m["Annual Vol"]) else "—")
col4.metric("Sharpe", f"{m['Sharpe']:.2f}" if pd.notna(m["Sharpe"]) else "—")
col5.metric("Max Drawdown", f"{m['Max Drawdown']*100:,.2f}%" if pd.notna(m["Max Drawdown"]) else "—")

st.markdown("<h2 style='font-size:34px; margin-top:22px;'>Performance</h2>", unsafe_allow_html=True)

# ---------------- Performance chart ----------------
perf_df = pd.DataFrame({"Portfolio": equity})
if bmk_equity is not None:
    # align equities
    bmk_equity = bmk_equity.reindex(equity.index).dropna()
    perf_df = perf_df.loc[bmk_equity.index]
    perf_df[bmk] = bmk_equity

fig_perf = px.line(
    perf_df.reset_index().rename(columns={"index": "date"}).melt(id_vars="date", var_name="Series", value_name="Value"),
    x="date",
    y="Value",
    color="Series",
    template="plotly_white",
)
fig_perf.update_layout(height=460, margin=dict(l=20, r=20, t=10, b=40), legend_title_text="")
fig_perf.update_xaxes(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
fig_perf.update_yaxes(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
st.plotly_chart(fig_perf, use_container_width=True)

# ---------------- Allocation + Correlation + Drawdown ----------------
cA, cB = st.columns([1, 1])

with cA:
    st.markdown("<h2 style='font-size:34px; margin-top:10px;'>Allocation</h2>", unsafe_allow_html=True)
    alloc_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    fig_alloc = px.pie(alloc_df, names="Ticker", values="Weight", hole=0.45, template="plotly_white")
    fig_alloc.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    st.plotly_chart(fig_alloc, use_container_width=True)

with cB:
    st.markdown("<h2 style='font-size:34px; margin-top:10px;'>Correlation</h2>", unsafe_allow_html=True)
    corr = rets.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        template="plotly_white",
    )
    fig_corr.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("<h2 style='font-size:34px; margin-top:18px;'>Drawdown</h2>", unsafe_allow_html=True)
dd = equity / equity.cummax() - 1
dd_df = pd.DataFrame({"date": dd.index, "drawdown": dd.values})

fig_dd = px.area(dd_df, x="date", y="drawdown", template="plotly_white")
fig_dd.update_layout(height=340, margin=dict(l=20, r=20, t=10, b=40))
fig_dd.update_xaxes(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
fig_dd.update_yaxes(title=None, tickformat=".0%", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
st.plotly_chart(fig_dd, use_container_width=True)

# ---------------- Table: returns & stats ----------------
st.markdown("<h2 style='font-size:34px; margin-top:18px;'>Assets</h2>", unsafe_allow_html=True)

asset_stats = []
ann_factor = 252
for t in tickers:
    r = rets[t]
    equity_t = (1 + r).cumprod()
    peak_t = equity_t.cummax()
    dd_t = (equity_t / peak_t - 1).min()
    asset_stats.append(
        {
            "Ticker": t,
            "Weight": float(weights[tickers.index(t)]),
            "Ann Return": float(r.mean() * ann_factor),
            "Ann Vol": float(r.std(ddof=0) * np.sqrt(ann_factor)),
            "Max DD": float(dd_t),
        }
    )

asset_df = pd.DataFrame(asset_stats)
asset_df["Weight"] = asset_df["Weight"].map(lambda x: f"{x:.2%}")
asset_df["Ann Return"] = asset_df["Ann Return"].map(lambda x: f"{x:.2%}")
asset_df["Ann Vol"] = asset_df["Ann Vol"].map(lambda x: f"{x:.2%}")
asset_df["Max DD"] = asset_df["Max DD"].map(lambda x: f"{x:.2%}")

st.dataframe(asset_df, use_container_width=True)

# ---------------- Download CSV ----------------
st.markdown("---")
out = pd.DataFrame({"Portfolio_Return": port_ret})
out["Portfolio_Equity"] = equity
if bmk_ret is not None:
    out[f"{bmk}_Return"] = bmk_ret.reindex(out.index)
    if bmk_equity is not None:
        out[f"{bmk}_Equity"] = bmk_equity.reindex(out.index)

csv = out.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv, file_name="portfolio_results.csv", mime="text/csv")
