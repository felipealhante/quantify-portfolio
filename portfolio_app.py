import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="Asset Allocation Pie", layout="centered")

st.title("Asset Allocation (Yahoo Tickers)")
st.caption("Pick tickers, assign weights (optional), and generate a pie chart.")

def normalize_tickers(raw: str) -> list[str]:
    parts = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        t = chunk.strip().upper()
        if t:
            parts.append(t)
    # unique but keep order
    seen, out = set(), []
    for t in parts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def parse_weights(raw: str, n: int):
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
        if np.any(w < 0) or w.sum() <= 0:
            return None
        return w / w.sum()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def validate_tickers(tickers: list[str]) -> dict:
    """
    Returns {ticker: True/False} by attempting to fetch quick info.
    This is best-effort validation; if Yahoo blocks, some may show False.
    """
    results = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info  # lightweight
            # if we can access last_price or currency, it likely exists
            _ = info.get("last_price", None)
            results[t] = True
        except Exception:
            results[t] = False
    return results

tickers_raw = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, NVDA", height=80)
tickers = normalize_tickers(tickers_raw)

weights_raw = st.text_input("Weights (optional, same count as tickers) e.g. 0.5,0.3,0.2", "")

c1, c2 = st.columns(2)
with c1:
    do_validate = st.checkbox("Validate tickers using Yahoo (optional)", value=True)
with c2:
    show_table = st.checkbox("Show table", value=True)

if st.button("Generate", type="primary", use_container_width=True):
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    weights = parse_weights(weights_raw, len(tickers))
    if weights is None:
        if weights_raw.strip():
            st.error("Weights invalid. Make sure you provide the same number as tickers, non-negative, and not all zero.")
            st.stop()
        weights = np.ones(len(tickers)) / len(tickers)

    # Optional validation
    invalid = []
    if do_validate:
        with st.spinner("Validating tickers with Yahoo..."):
            status = validate_tickers(tickers)
        invalid = [t for t, ok in status.items() if not ok]
        if invalid:
            st.warning(f"Some tickers may be invalid or Yahoo blocked validation: {', '.join(invalid)}")

    alloc_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    alloc_df["Weight %"] = (alloc_df["Weight"] * 100).round(2)

    fig = px.pie(
        alloc_df,
        names="Ticker",
        values="Weight",
        hole=0.45,
        title="Asset Allocation",
        template="plotly_white",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if show_table:
        st.subheader("Allocation Table")
        st.dataframe(
            alloc_df[["Ticker", "Weight %"]].sort_values("Weight %", ascending=False),
            use_container_width=True,
        )
