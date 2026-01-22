# arima_app.py  (Streamlit + Plotly, fixed chart rendering)

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARIMA Forecast + Backtest", layout="wide")
st.title("ARIMA Forecast + Walk-Forward Backtest (Interactive)")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def get_price_series(ticker: str, years: int) -> pd.Series:
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=365 * years)

    # auto_adjust=True => adjusted close is "Close"
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker.")

    if "Close" not in df.columns:
        raise ValueError("Yahoo response missing 'Close' column.")

    price = df["Close"].dropna()
    price.index = pd.to_datetime(price.index).tz_localize(None)
    price = price.asfreq("B").ffill()
    price = price.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    price.name = "price"
    return price


def adf_pvalue(series: pd.Series) -> float:
    _, pvalue, *_ = adfuller(series.dropna())
    return float(pvalue)


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def walk_forward_backtest_arima(
    returns: pd.Series,
    order=(1, 0, 1),
    initial_train: int = 500,
    horizon: int = 5,
    step: int = 5,
):
    returns = returns.dropna().astype(float)

    if initial_train < 100:
        raise ValueError("Initial train too small; try 300–800 for daily data.")
    if initial_train + horizon >= len(returns):
        raise ValueError("Not enough data for initial_train + horizon.")

    preds, actuals = [], []
    i = initial_train

    while i + horizon <= len(returns):
        train = returns.iloc[:i]
        test = returns.iloc[i : i + horizon]

        try:
            res = ARIMA(train, order=order).fit()
            fc = res.get_forecast(steps=horizon).predicted_mean
            fc = pd.Series(fc, index=test.index).astype(float)
        except Exception:
            i += step
            continue

        preds.append(fc)
        actuals.append(test)
        i += step

    if not preds:
        raise ValueError("Backtest failed on all windows. Try different (p,d,q) or more history.")

    pred = pd.concat(preds).sort_index()
    act = pd.concat(actuals).sort_index()
    pred, act = pred.align(act, join="inner")
    return pred, act


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", value="^GSPC").strip().upper()
years = st.sidebar.slider("Years of history", 1, 15, 5)

st.sidebar.subheader("ARIMA(p,d,q)")
p = st.sidebar.number_input("p", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("d", min_value=0, max_value=2, value=0, step=1)
q = st.sidebar.number_input("q", min_value=0, max_value=5, value=1, step=1)

forecast_days = st.sidebar.slider("Forecast business days ahead", 1, 60, 20)

st.sidebar.subheader("Backtest settings")
initial_train = st.sidebar.number_input("Initial train size (days)", min_value=100, max_value=3000, value=500, step=50)
horizon = st.sidebar.number_input("Forecast horizon per step (days)", min_value=1, max_value=60, value=5, step=1)
step = st.sidebar.number_input("Step size (days)", min_value=1, max_value=60, value=5, step=1)

show_debug = st.sidebar.checkbox("Show debug", value=False)

run = st.sidebar.button("Run", type="primary", use_container_width=True)
if not run:
    st.stop()

# ----------------------------
# Data
# ----------------------------
try:
    price = get_price_series(ticker, years)
except Exception as e:
    st.error(str(e))
    st.stop()

returns = np.log(price / price.shift(1)).dropna().astype(float)

c1, c2, c3 = st.columns(3)
c1.metric("Observations (prices)", f"{len(price):,}")
c2.metric("Observations (returns)", f"{len(returns):,}")
c3.metric("ADF p-value (returns)", f"{adf_pvalue(returns):.4f}")

# ----------------------------
# Fit + Forecast (returns -> prices)
# ----------------------------
try:
    res = ARIMA(returns, order=(int(p), int(d), int(q))).fit()
except Exception as e:
    st.error(f"ARIMA fit failed: {e}")
    st.stop()

fc_ret = res.get_forecast(steps=int(forecast_days)).predicted_mean
fc_ret = pd.Series(fc_ret).astype(float).replace([np.inf, -np.inf], np.nan).dropna()

last_price = float(price.iloc[-1])
fc_index = pd.bdate_range(start=price.index[-1] + pd.Timedelta(days=1), periods=len(fc_ret))
fc_price = last_price * np.exp(fc_ret.cumsum())
fc_price = pd.Series(fc_price.values, index=fc_index, name="Forecast").astype(float)
fc_price.index = pd.to_datetime(fc_price.index).tz_localize(None)
fc_price = fc_price.replace([np.inf, -np.inf], np.nan).dropna()

# ----------------------------
# Forecast chart (Plotly, robust)
# ----------------------------
st.subheader("Forecast (Interactive)")

hist_window = min(len(price), 252)
hist = price.iloc[-hist_window:].copy()
hist.index = pd.to_datetime(hist.index).tz_localize(None)
hist = hist.replace([np.inf, -np.inf], np.nan).dropna()

plot_df = pd.concat(
    [hist.rename("Historical"), fc_price.rename("Forecast")],
    axis=1
).reset_index().rename(columns={"index": "Date"})

long_df = plot_df.melt(id_vars="Date", var_name="Series", value_name="Price").dropna()

fig_forecast = px.line(
    long_df,
    x="Date",
    y="Price",
    color="Series",
    template="plotly_white",
    title=f"{ticker} ARIMA({p},{d},{q}) Forecast ({forecast_days} business days)",
)
fig_forecast.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=30))
fig_forecast.update_xaxes(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
fig_forecast.update_yaxes(title="Price", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
st.plotly_chart(fig_forecast, use_container_width=True)

st.write(f"Last historical price: **{last_price:.2f}**")
st.write(f"Last forecast price: **{float(fc_price.iloc[-1]):.2f}**")

with st.expander("ARIMA Summary"):
    st.text(res.summary())

if show_debug:
    with st.expander("Debug (data checks)"):
        st.write("Hist points:", len(hist), "Forecast points:", len(fc_price))
        st.write("Hist date range:", hist.index.min(), "→", hist.index.max())
        st.write("Forecast date range:", fc_price.index.min(), "→", fc_price.index.max())
        st.dataframe(long_df.tail(20), use_container_width=True)

# ----------------------------
# Backtest (returns)
# ----------------------------
st.subheader("Walk-forward Backtest (Interactive)")

try:
    pred_ret, act_ret = walk_forward_backtest_arima(
        returns,
        order=(int(p), int(d), int(q)),
        initial_train=int(initial_train),
        horizon=int(horizon),
        step=int(step),
    )
except Exception as e:
    st.error(f"Backtest failed: {e}")
    st.stop()

# Metrics
ret_mae = mae(act_ret, pred_ret)
ret_rmse = rmse(act_ret, pred_ret)
direction_acc = float((np.sign(act_ret.values) == np.sign(pred_ret.values)).mean())

m1, m2, m3 = st.columns(3)
m1.metric("Returns MAE", f"{ret_mae:.6f}")
m2.metric("Returns RMSE", f"{ret_rmse:.6f}")
m3.metric("Directional Accuracy", f"{direction_acc:.2%}")

bt_df = pd.DataFrame({"Actual": act_ret, "Predicted": pred_ret}).dropna().reset_index().rename(columns={"index": "Date"})
bt_long = bt_df.melt(id_vars="Date", var_name="Series", value_name="Log Return")

fig_bt = px.line(
    bt_long,
    x="Date",
    y="Log Return",
    color="Series",
    template="plotly_white",
    title="Walk-forward backtest (returns)",
)
fig_bt.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=30))
fig_bt.update_xaxes(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
fig_bt.update_yaxes(title="Log return", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
st.plotly_chart(fig_bt, use_container_width=True)

with st.expander("Backtest data (tail)"):
    st.dataframe(bt_df.tail(50), use_container_width=True)
