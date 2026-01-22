import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARIMA Stock Forecast", layout="wide")
st.title("ARIMA Stock Forecast + Walk-Forward Backtest")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def get_price_series(ticker: str, years: int) -> pd.Series:
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=365 * years)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or connection.")

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = df[col].dropna()

    price.index = pd.to_datetime(price.index)
    price = price.asfreq("B").ffill()  # business days + fill gaps
    price.name = "price"
    return price.astype(float)


def adf_pvalue(series: pd.Series) -> float:
    stat, pvalue, *_ = adfuller(series.dropna())
    return float(pvalue)


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def returns_to_prices(start_price: float, returns: pd.Series) -> pd.Series:
    return float(start_price) * np.exp(returns.cumsum())


def walk_forward_backtest_arima(
    returns: pd.Series,
    order=(1, 0, 1),
    initial_train: int = 500,
    horizon: int = 5,
    step: int = 5,
):
    returns = returns.dropna().astype(float)

    if initial_train < 50:
        raise ValueError("initial_train too small; use at least ~200-500 for daily data.")
    if initial_train + horizon >= len(returns):
        raise ValueError("Not enough data for given initial_train and horizon.")

    preds, actuals = [], []
    i = initial_train

    while i + horizon <= len(returns):
        train = returns.iloc[:i]
        test = returns.iloc[i:i + horizon]

        try:
            res = ARIMA(train, order=order).fit()
            fc = res.get_forecast(steps=horizon).predicted_mean
            fc = pd.Series(fc, index=test.index).astype(float)
        except Exception:
            # skip failed window
            i += step
            continue

        preds.append(fc)
        actuals.append(test)
        i += step

    if not preds:
        raise ValueError("Backtest failed on all windows. Try different (p,d,q) or larger history.")

    pred = pd.concat(preds).sort_index()
    act = pd.concat(actuals).sort_index()
    pred, act = pred.align(act, join="inner")
    return pred, act


# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
years = st.sidebar.slider("Years of history", 1, 15, 5)

st.sidebar.subheader("ARIMA order (p,d,q)")
p = st.sidebar.number_input("p", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("d", min_value=0, max_value=2, value=0, step=1)
q = st.sidebar.number_input("q", min_value=0, max_value=5, value=1, step=1)

forecast_days = st.sidebar.slider("Forecast business days ahead", 1, 60, 20)

st.sidebar.subheader("Backtest settings")
initial_train = st.sidebar.number_input("Initial train size (days)", min_value=100, max_value=3000, value=500, step=50)
horizon = st.sidebar.number_input("Forecast horizon per step (days)", min_value=1, max_value=60, value=5, step=1)
step = st.sidebar.number_input("Step size (days)", min_value=1, max_value=60, value=5, step=1)

run = st.sidebar.button("Run", type="primary", use_container_width=True)

if not run:
    st.stop()

# ----------------------------
# Run model
# ----------------------------
try:
    price = get_price_series(ticker, years=years)
except Exception as e:
    st.error(str(e))
    st.stop()

returns = np.log(price / price.shift(1)).dropna()

pval = adf_pvalue(returns)
colA, colB, colC = st.columns(3)
colA.metric("Obs (prices)", f"{len(price):,}")
colB.metric("Obs (returns)", f"{len(returns):,}")
colC.metric("ADF p-value (returns)", f"{pval:.4f}")

if len(returns) < 150:
    st.warning("Not much data. Increase years for better ARIMA stability.")

# ----------------------------
# Forecast
# ----------------------------
st.subheader("Forecast")

try:
    model = ARIMA(returns, order=(int(p), int(d), int(q)))
    res = model.fit()
except Exception as e:
    st.error(f"ARIMA fit failed: {e}")
    st.stop()

# forecast returns
fc_ret = res.get_forecast(steps=int(forecast_days)).predicted_mean
fc_ret = pd.Series(fc_ret).iloc[: int(forecast_days)].astype(float)

# returns -> prices
last_price = float(price.iloc[-1])
forecast_index = pd.bdate_range(start=price.index[-1] + pd.Timedelta(days=1), periods=len(fc_ret))
fc_prices = last_price * np.exp(fc_ret.cumsum())
fc_prices = pd.Series(fc_prices.values, index=forecast_index, name="forecast")

# plot: last 252 business days + forecast
hist_window = min(len(price), 252)
hist = price.iloc[-hist_window:]

fig1 = plt.figure(figsize=(12, 5))
plt.plot(hist.index, hist.values, label="Historical", linewidth=2)
plt.plot(fc_prices.index, fc_prices.values, label="Forecast", linestyle="--", linewidth=2)
plt.title(f"{ticker} ARIMA({p},{d},{q}) Forecast ({forecast_days} business days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(alpha=0.3)
plt.legend()
st.pyplot(fig1)

st.write(f"Last historical price: **{last_price:.2f}**")
st.write(f"Last forecast price: **{float(fc_prices.iloc[-1]):.2f}**")

with st.expander("ARIMA Summary"):
    st.text(res.summary())

# ----------------------------
# Backtest
# ----------------------------
st.subheader("Walk-forward Backtest")

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

# metrics in returns space
ret_mae = mae(act_ret, pred_ret)
ret_rmse = rmse(act_ret, pred_ret)
direction_acc = float((np.sign(act_ret.values) == np.sign(pred_ret.values)).mean())

# convert to prices
first_dt = pred_ret.index.min()
prev_dt = first_dt - pd.tseries.offsets.BDay(1)

if prev_dt not in price.index:
    # fallback: use last price before first_dt
    prev_dt = price.index[price.index < first_dt][-1]

start_price = float(price.loc[prev_dt])

pred_price = returns_to_prices(start_price, pred_ret)
pred_price.index = pred_ret.index
act_price = price.loc[pred_price.index]

px_mae = mae(act_price, pred_price)
px_rmse = rmse(act_price, pred_price)

m1, m2, m3 = st.columns(3)
m1.metric("Returns MAE", f"{ret_mae:.6f}")
m2.metric("Returns RMSE", f"{ret_rmse:.6f}")
m3.metric("Directional Acc.", f"{direction_acc:.2%}")

m4, m5 = st.columns(2)
m4.metric("Price MAE", f"{px_mae:.2f}")
m5.metric("Price RMSE", f"{px_rmse:.2f}")

fig2 = plt.figure(figsize=(12, 5))
plt.plot(act_price.index, act_price.values, label="Actual Price", linewidth=2)
plt.plot(pred_price.index, pred_price.values, label="Predicted Price (walk-forward)", linestyle="--", linewidth=2)
plt.title(f"{ticker} ARIMA({p},{d},{q}) Walk-forward Backtest")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(alpha=0.3)
plt.legend()
st.pyplot(fig2)

with st.expander("Backtest data (preview)"):
    out = pd.DataFrame({"actual_return": act_ret, "pred_return": pred_ret})
    st.dataframe(out.tail(50), use_container_width=True)
