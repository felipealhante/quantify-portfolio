import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Ticker Selector")

ticker_input = st.text_input(
    "Enter Yahoo tickers (comma separated)",
    value="AAPL,^GSPC,GOOGL"
)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=4)

if tickers:
    data = yf.download(tickers, period=period, progress=False)

    if data.empty:
        st.error("No data found. Check ticker symbols.")
    else:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
        else:
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            prices = data[[col]].rename(columns={col: tickers[0]})

        st.subheader("Normalized Prices (start = 100)")
        norm = 100 * prices / prices.iloc[0]
        st.line_chart(norm)

        st.subheader("Last Prices")
        st.dataframe(prices.tail())
else:
    st.info("Enter at least one ticker.")
