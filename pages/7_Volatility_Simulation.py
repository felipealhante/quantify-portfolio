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


def annualize_vol(daily_vol: pd.Series, trading_days: int = 252) -> pd.Series:
    return daily_vol * np.sqrt(trading_days)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    r = np.log(prices / prices.shift(1)).dropna()
    r.name = "log_return"
    return r


def rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std()


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    EWMA volatility (RiskMetrics):
    sigma_t^2 = lam*sigma_{t-1}^2 + (1-lam)*r_{t-1}^2
    """
    r2 = (returns ** 2).values
    sigma2 = np.zeros_like(r2)
    sigma2[0] = r2[0]
    for i in range(1, len(r2)):
        sigma2[i] = lam * sigma2[i - 1] + (1 - lam) * r2[i - 1]
    out = pd.Series(np.sqrt(sigma2), index=returns.index, name="ewma_vol_daily")
    return out


def try_garch_vol(returns: pd.Series):
    """
    Optional GARCH(1,1) using 'arch'. Returns:
    - cond_vol_ann: pd.Series or None
    - next_vol_ann: float or None
    - err: str or None
    """
    try:
        from arch import arch_model
    except Exception:
        return None, None, "Package 'arch' not installed. Add it to requirements.txt: arch"

    # percent returns for stability
    r_pct = returns * 100.0
    am = arch_model(r_pct, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")

    try:
        res = am.fit(disp="off")
    except Exception as e:
        return None, None, f"GARCH fit failed: {e}"

    # conditional volatility is in percent units
    cond_vol_daily = (res.conditional_volatility / 100.0).astype(float)
    cond_vol_ann = annualize_vol(pd.Series(cond_vol_daily, index=returns.index, name="garch_vol_ann"))

    # 1-step forecast variance is percent^2
    f = res.forecast(horizon=1, reindex=False)
    var_next_pct2 = float(f.variance.values[-1, 0])
    vol_next_daily = np.sqrt(var_next_pct2) / 100.0
    vol_next_ann = float(annualize_vol(pd.Series([vol_next_daily])).iloc[0])

    return cond_vol_ann, vol_next_ann, None


@st.cache_data(show_spinner=False)
def download_prices_multi_safe(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    """
    Downloads Close (or Adj Close) and handles yfinance MultiIndex columns cleanly.
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
        raise ValueError(f"{ticker}: no valid prices after cleaning.")
    return s


# -----------------------------
# Page UI
# -----------------------------
st.set_page_config(page_title="Volatility Modeling", layout="wide")
st.title("Volatility Modeling")
st.caption("Rolling historical vol • EWMA (RiskMetrics) • optional GARCH(1,1). Annualized for readability.")

with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Ticker", value="NVDA")
    years = st.number_input("Years of history", min_value=1, max_value=30, value=5, step=1)
    window = st.number_input("Rolling window (days)", min_value=5, max_value=252, value=20, step=1)
    lam = st.slider("EWMA lambda (λ)", min_value=0.70, max_value=0.99, value=0.94, step=0.01)

    use_garch = st.checkbox("Enable GARCH(1,1) (requires `arch`)", value=False)

    run = st.button("Run", type="primary", use_container_width=True)

if not run:
    st.info("Set your inputs on the left and click **Run**.")
    st.stop()

ticker = normalize_ticker(ticker)
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=365 * int(years))

# -----------------------------
# Data + models
# -----------------------------
with st.spinner("Downloading prices and computing volatility..."):
    prices = download_prices_multi_safe(ticker, start_date, end_date)
    rets = compute_log_returns(prices)

    vol_hist_daily = rolling_vol(rets, window=int(window))
    vol_ewma_daily = ewma_vol(rets, lam=float(lam))

    vol_hist_ann = annualize_vol(vol_hist_daily).rename("hist_vol_ann")
    vol_ewma_ann = annualize_vol(vol_ewma_daily).rename("ewma_vol_ann")

    garch_ann, garch_next_ann, garch_err = (None, None, None)
    if use_garch:
        garch_ann, garch_next_ann, garch_err = try_garch_vol(rets)

# Next-day (proxy = last available)
vol_hist_next_ann = float(vol_hist_ann.dropna().iloc[-1]) if not vol_hist_ann.dropna().empty else np.nan
vol_ewma_next_ann = float(vol_ewma_ann.dropna().iloc[-1]) if not vol_ewma_ann.dropna().empty else np.nan

# -----------------------------
# KPI Row
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric(f"Hist vol (last, {int(window)}d)", f"{vol_hist_next_ann:.2%}" if np.isfinite(vol_hist_next_ann) else "—")
c3.metric(f"EWMA vol (last, λ={lam:.2f})", f"{vol_ewma_next_ann:.2%}" if np.isfinite(vol_ewma_next_ann) else "—")
if use_garch:
    if garch_err:
        c4.metric("GARCH next-day vol", "—")
    else:
        c4.metric("GARCH next-day vol", f"{garch_next_ann:.2%}")
else:
    c4.metric("GARCH next-day vol", "Off")

if use_garch and garch_err:
    st.warning(garch_err)

# -----------------------------
# Time series chart
# -----------------------------
st.subheader("Annualized volatility over time")

df_vol = pd.DataFrame(index=rets.index)
df_vol["Rolling Hist Vol (ann.)"] = vol_hist_ann
df_vol["EWMA Vol (ann.)"] = vol_ewma_ann
if use_garch and (garch_ann is not None) and (not garch_ann.empty):
    df_vol["GARCH(1,1) Vol (ann.)"] = garch_ann

df_vol = df_vol.dropna(how="all")

fig = go.Figure()
for col in df_vol.columns:
    fig.add_trace(go.Scatter(x=df_vol.index, y=df_vol[col], mode="lines", name=col))

fig.update_layout(
    template="plotly_dark",
    height=520,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_title="Date",
    yaxis_title="Annualized volatility",
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Returns distribution
# -----------------------------
st.subheader("Daily log-return distribution")

ret_df = pd.DataFrame({"log_return": rets.values})
fig2 = px.histogram(ret_df, x="log_return", nbins=80, title="Daily log returns")
fig2.update_layout(
    template="plotly_dark",
    height=380,
    margin=dict(l=10, r=10, t=60, b=10),
    xaxis_title="Daily log return",
    yaxis_title="Count",
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Return stats (like your print section)
# -----------------------------
st.subheader("Basic return stats")

mean_r = float(rets.mean())
std_r = float(rets.std())
skew_r = float(rets.skew())
kurt_r = float(rets.kurtosis())

s1, s2, s3, s4 = st.columns(4)
s1.metric("Mean daily log return", f"{mean_r:.6f}")
s2.metric("Std daily log return", f"{std_r:.6f}")
s3.metric("Skew", f"{skew_r:.3f}")
s4.metric("Kurtosis", f"{kurt_r:.3f}")

with st.expander("Show data"):
    st.write("Prices:")
    st.dataframe(prices.to_frame("Price").tail(30), use_container_width=True)
    st.write("Volatility table:")
    st.dataframe(df_vol.tail(30), use_container_width=True)
