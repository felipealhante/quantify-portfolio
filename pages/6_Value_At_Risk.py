import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# -----------------------------
# Helpers
# -----------------------------
def normalize_ticker(t: str) -> str:
    return (t or "").strip().upper().replace("ˆ", "^")


@st.cache_data(show_spinner=False)
def download_close_multi_safe(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    """
    Returns a 1D Series of Close (or Adj Close) and handles yfinance MultiIndex columns.
    """
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
        raise ValueError(f"No data returned for {ticker}.")

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # MultiIndex case: columns like ('Close','AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            close_cols = [c for c in df.columns if c[0] in ("Adj Close", "Close")]
            if not close_cols:
                raise ValueError(f"{ticker}: missing Close/Adj Close in MultiIndex columns.")
            s = df[close_cols[0]]
    else:
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            raise ValueError(f"{ticker}: missing Close/Adj Close. Columns: {list(df.columns)}")
        s = df[col]

    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    s.name = ticker
    if s.empty:
        raise ValueError(f"{ticker}: Close series empty after cleaning.")
    return s


def historical_var(dollar_returns: pd.Series, confidence: float) -> float:
    """
    Historical VaR as positive dollars (loss amount).
    """
    alpha = (1 - confidence) * 100
    return float(-np.percentile(dollar_returns, alpha))


# -----------------------------
# Page config / style
# -----------------------------
st.set_page_config(page_title="Portfolio VaR (Historical)", layout="wide")

st.title("Portfolio Value-at-Risk")
st.caption("Measure the risk of your portfolio.")


# -----------------------------
# Sidebar Inputs
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    years = st.number_input("How many years back?", min_value=1, max_value=30, value=3, step=1)
    portfolio_value = st.number_input("Portfolio value ($)", min_value=1.0, value=10000.0, step=500.0)
    days = st.number_input("Rolling window (days)", min_value=1, max_value=252, value=20, step=1)
    confidence_interval = st.slider("Confidence interval", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        help="Example: NVDA, KC=F, AAPL",)

    run = st.button("Run", type="primary", use_container_width=True)


if not run:
    st.info("Set your inputs on the left and click **Run**.")
    st.stop()


# -----------------------------
# Validate / parse
# -----------------------------
tickers = [normalize_ticker(t) for t in (tickers_input or "").split(",") if t.strip()]
tickers = [t for t in tickers if t]

if len(tickers) == 0:
    st.error("You must enter at least one ticker.")
    st.stop()

end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=365 * int(years))


# -----------------------------
# Download prices
# -----------------------------
with st.spinner("Downloading prices..."):
    prices = {}
    skipped = []

    for t in tickers:
        try:
            prices[t] = download_close_multi_safe(t, start_date, end_date)
        except Exception as e:
            skipped.append((t, str(e)))

if len(prices) == 0:
    st.error("No valid tickers returned data. Try different symbols or a shorter date range.")
    st.stop()

prices_df = pd.concat(prices.values(), axis=1).dropna()

if prices_df.empty:
    st.error("No overlapping dates between the tickers. Try fewer years or fewer tickers.")
    if skipped:
        with st.expander("Download errors (skipped tickers)"):
            for t, msg in skipped:
                st.write(f"- **{t}**: {msg}")
    st.stop()


# -----------------------------
# Returns + rolling window in dollars
# -----------------------------
log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

weights = np.array([1.0 / len(prices_df.columns)] * len(prices_df.columns))
portfolio_returns = (log_returns * weights).sum(axis=1)

range_returns = portfolio_returns.rolling(window=int(days)).sum().dropna()
range_returns_dollar = range_returns * float(portfolio_value)

if range_returns_dollar.empty:
    st.error("Not enough data for that rolling window. Reduce 'days' or increase 'years'.")
    st.stop()

VaR = historical_var(range_returns_dollar, float(confidence_interval))


# -----------------------------
# Top KPIs
# -----------------------------
used_tickers = list(prices_df.columns)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers used", f"{len(used_tickers)}")
c2.metric("Window", f"{int(days)} days")
c3.metric("Confidence", f"{int(confidence_interval*100)}%")
c4.metric("VaR (loss)", f"${VaR:,.2f}")

if skipped:
    st.warning(f"Some tickers were skipped: {', '.join([t for t, _ in skipped])}")
    with st.expander("Why some tickers were skipped"):
        for t, msg in skipped:
            st.write(f"- **{t}**: {msg}")


# -----------------------------
# Histogram (Plotly) + VaR line
# -----------------------------
st.subheader("Distribution of rolling portfolio returns ($)")

hist_df = pd.DataFrame({"RollingReturn_$": range_returns_dollar.values})

fig = px.histogram(
    hist_df,
    x="RollingReturn_$",
    nbins=60,
    title=f"Distribution of {int(days)}-Day Portfolio Returns ($) — {', '.join(used_tickers)}",
)

# VaR line (VaR is positive loss; threshold is -VaR)
var_x = -VaR
fig.add_vline(
    x=var_x,
    line_dash="dash",
    annotation_text=f"VaR ({int(confidence_interval*100)}%): {var_x:,.0f}",
    annotation_position="top right",
)

fig.update_layout(
    template="plotly_dark",
    height=520,
    margin=dict(l=10, r=10, t=60, b=10),
    xaxis_title="Rolling portfolio return ($)",
    yaxis_title="Count",
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Extra: show the time-series of rolling $ returns (matches your “app vibe”)
# -----------------------------
st.subheader("Rolling return over time ($)")

ts_df = pd.DataFrame({"Date": range_returns_dollar.index, "RollingReturn_$": range_returns_dollar.values})
fig2 = px.line(ts_df, x="Date", y="RollingReturn_$", title=f"{int(days)}-Day Rolling Return ($)")
fig2.add_hline(y=0, line_dash="dot")
fig2.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# Data preview
# -----------------------------
with st.expander("Show data"):
    st.write("Prices (aligned):")
    st.dataframe(prices_df.tail(20), use_container_width=True)

    st.write("Rolling dollar returns:")
    st.dataframe(ts_df.tail(20), use_container_width=True)
