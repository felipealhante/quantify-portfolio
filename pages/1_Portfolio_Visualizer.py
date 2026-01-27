import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import requests
from urllib.parse import quote

if "ran" not in st.session_state:
    st.session_state.ran = False

# =========================================================
# CONFIG: asset-type overrides (so we don't show "Unknown")
# =========================================================
ASSET_OVERRIDES = {
    # Commodities / futures (Yahoo often returns Unknown)
    "CL=F": {"Asset Type": "Commodity (Futures)", "Sector": "Commodities", "Industry": "Energy - Oil"},
    "NG=F": {"Asset Type": "Commodity (Futures)", "Sector": "Commodities", "Industry": "Energy - Natural Gas"},
    "GC=F": {"Asset Type": "Commodity (Futures)", "Sector": "Commodities", "Industry": "Precious Metals"},
    "SI=F": {"Asset Type": "Commodity (Futures)", "Sector": "Commodities", "Industry": "Precious Metals"},
    # Indices / FX examples
    "^GSPC": {"Asset Type": "Index", "Sector": "Index", "Industry": "Index"},
    "^IXIC": {"Asset Type": "Index", "Sector": "Index", "Industry": "Index"},
    "EURUSD=X": {"Asset Type": "FX", "Sector": "FX", "Industry": "FX"},
}

# Contract multipliers (fix P/L for futures)
DEFAULT_MULTIPLIERS = {
    "CL=F": 1000,   # WTI: 1 contract = 1000 barrels
    "NG=F": 10000,  # Henry Hub: often 10,000 MMBtu (Yahoo symbol = NG=F)
    "GC=F": 100,    # Gold: 100 oz
    "SI=F": 5000,   # Silver: 5,000 oz
}

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Portfolio Manager", layout="wide")
st.title("Portfolio Manager")
st.caption("Visualize your portfolio in a multitude of ways.")

# =========================================================
# HELPERS
# =========================================================
def normalize_tickers(raw: str) -> list[str]:
    return list(dict.fromkeys(
        t.strip().upper().replace("ˆ", "^")
        for t in raw.replace("\n", ",").replace(" ", ",").split(",")
        if t.strip()
    ))

@st.cache_data(show_spinner=False)
def download_prices(tickers, start, end):
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()
        close = df["Close"].copy()
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.ffill().dropna(how="all")
    close.columns = close.columns.astype(str)
    return close

@st.cache_data(show_spinner=False)
def latest_prices(tickers):
    end = dt.date.today()
    start = end - dt.timedelta(days=30)
    px_df = download_prices(tickers, start, end)
    if px_df.empty:
        return pd.Series(dtype=float)
    last = px_df.iloc[-1].dropna()
    last.index = last.index.astype(str)
    return last.astype(float)

def infer_asset_type(ticker: str) -> str:
    t = str(ticker).upper().strip()
    if t in ASSET_OVERRIDES:
        return ASSET_OVERRIDES[t]["Asset Type"]
    if t.startswith("^"):
        return "Index"
    if t.endswith("=X"):
        return "FX"
    if t.endswith("=F") or t in DEFAULT_MULTIPLIERS:
        return "Commodity (Futures)"
    if t.endswith("-USD"):
        return "Crypto"
    return "Equity"

@st.cache_data(show_spinner=False)
def get_sector_industry(tickers):
    """
    Returns Sector/Industry/Asset Type. If Yahoo gives missing/Unknown, we replace with asset-type defaults.
    """
    rows = []
    for t in tickers:
        t = str(t).strip().upper()
        if not t:
            continue

        # Overrides first
        if t in ASSET_OVERRIDES:
            o = ASSET_OVERRIDES[t]
            rows.append({"Ticker": t, "Asset Type": o["Asset Type"], "Sector": o["Sector"], "Industry": o["Industry"]})
            continue

        asset_type = infer_asset_type(t)

        # Try Yahoo info (best-effort)
        sector = None
        industry = None
        try:
            info = yf.Ticker(t).info or {}
            sector = info.get("sector")
            industry = info.get("industry")
        except Exception:
            pass

        # Replace missing/unknown with meaningful labels
        if not sector or str(sector).strip().lower() == "unknown":
            sector = asset_type
        if not industry or str(industry).strip().lower() == "unknown":
            industry = asset_type

        rows.append({"Ticker": t, "Asset Type": asset_type, "Sector": sector, "Industry": industry})

    if not rows:
        return pd.DataFrame(columns=["Asset Type", "Sector", "Industry"]).set_index(pd.Index([], name="Ticker"))

    return pd.DataFrame(rows).set_index("Ticker")

@st.cache_data(show_spinner=False, ttl=600)
def yahoo_news(ticker, n=10):
    t = str(ticker).strip().upper()
    try:
        items = yf.Ticker(t).news or []
    except Exception:
        items = []
    rows = []
    for it in items[:n]:
        ts = it.get("providerPublishTime")
        title = (it.get("title") or "").strip()
        link = (it.get("link") or "").strip()
        publisher = (it.get("publisher") or "").strip()
        when = pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M") if ts else ""
        rows.append({"time": when, "title": title, "publisher": publisher, "link": link})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=600)
def google_news(ticker, n=10):
    url = f"https://news.google.com/rss/search?q={quote(ticker + ' stock')}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        xml = r.text
    except Exception:
        return pd.DataFrame(columns=["time", "title", "publisher", "link"])

    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)

    rows = []
    for item in root.findall(".//item")[:n]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        publisher = (source_el.text.strip() if source_el is not None and source_el.text else "").strip()
        rows.append({"time": pub_date, "title": title, "publisher": publisher, "link": link})

    return pd.DataFrame(rows)

def render_news(df: pd.DataFrame, max_items: int):
    if df is None or df.empty:
        st.info("No news found.")
        return

    df = df.copy()
    for c in ["title", "link", "publisher", "time"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    df = df[df["title"] != ""]
    if df.empty:
        st.info("News returned, but titles were empty.")
        return

    for _, row in df.head(max_items).iterrows():
        title, link = row["title"], row["link"]
        pub = row["publisher"] or "Unknown source"
        when = row["time"]
        if link:
            st.markdown(f"- **[{title}]({link})**  \n  {pub} · {when}")
        else:
            st.markdown(f"- **{title}**  \n  {pub} · {when}")

# =========================================================
# SIDEBAR INPUTS
# =========================================================
with st.sidebar:
    st.header("Inputs")
    tickers = normalize_tickers(st.text_area("Tickers (comma separated)"))
    start_date = st.date_input("Performance start", dt.date.today() - dt.timedelta(days=365))
    end_date = st.date_input("Performance end", dt.date.today())
    show_pie = st.checkbox("Show exposure pie chart", True)
    show_debug = st.checkbox("Show debug", False)

if not tickers:
    st.info("Add at least one ticker.")
    st.stop()

if start_date >= end_date:
    st.error("Start must be before end.")
    st.stop()

# =========================================================
# TRADE INPUTS
# =========================================================
st.markdown("### Trade inputs")


base = pd.DataFrame({
    "Ticker": tickers,
    "Position": ["Long"] * len(tickers),
    "Entry Price": [np.nan] * len(tickers),
    "Shares": [1.0] * len(tickers),
    "Multiplier": [float(DEFAULT_MULTIPLIERS.get(t, 1.0)) for t in tickers],
})

edited = st.data_editor(
    base,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Position": st.column_config.SelectboxColumn(options=["Long", "Short"]),
        "Entry Price": st.column_config.NumberColumn(min_value=0.0, step=0.01),
        "Shares": st.column_config.NumberColumn(min_value=0.0, step=1.0),
        "Multiplier": st.column_config.NumberColumn(min_value=0.0, step=1.0, help="Stocks=1. Futures use contract multiplier."),
    },
)

run = st.button("Run", type="primary", use_container_width=True)

if run:
    st.session_state.ran = True

if not st.session_state.ran:
    st.stop()


inputs = edited.copy()
inputs["Ticker"] = inputs["Ticker"].astype(str).str.upper().str.strip()
inputs["Position"] = inputs["Position"].astype(str).str.strip()

inputs["Shares"] = pd.to_numeric(inputs["Shares"], errors="coerce").fillna(1.0)
inputs.loc[inputs["Shares"] <= 0, "Shares"] = 1.0

inputs["Multiplier"] = pd.to_numeric(inputs["Multiplier"], errors="coerce").fillna(1.0)
inputs.loc[inputs["Multiplier"] <= 0, "Multiplier"] = 1.0

inputs["Entry Price"] = pd.to_numeric(inputs["Entry Price"], errors="coerce")
if inputs["Entry Price"].isna().any():
    st.error("Fill all entry prices.")
    st.stop()

# =========================================================
# SNAPSHOT METRICS
# =========================================================
with st.spinner("Fetching latest prices..."):
    latest = latest_prices(inputs["Ticker"].tolist())

missing = [t for t in inputs["Ticker"] if t not in latest.index]
if missing:
    st.warning(f"Some tickers returned no recent price and will be skipped: {', '.join(missing)}")

inputs = inputs[inputs["Ticker"].isin(latest.index)].copy()
if inputs.empty:
    st.error("No usable tickers after fetching prices.")
    st.stop()

inputs["Last Price"] = inputs["Ticker"].map(latest.to_dict()).astype(float)

# Exposure and P/L use multiplier (FIXED)
inputs["Exposure"] = (inputs["Entry Price"] * inputs["Shares"] * inputs["Multiplier"]).abs()

inputs["P/L"] = np.where(
    inputs["Position"].str.lower() == "long",
    (inputs["Last Price"] - inputs["Entry Price"]) * inputs["Shares"] * inputs["Multiplier"],
    (inputs["Entry Price"] - inputs["Last Price"]) * inputs["Shares"] * inputs["Multiplier"],
)
inputs["P/L %"] = np.where(inputs["Exposure"] > 0, inputs["P/L"] / inputs["Exposure"], np.nan)

# Add Sector/Industry/Asset Type (NO UNKNOWN)
cls = get_sector_industry(inputs["Ticker"].tolist())
inputs = inputs.merge(cls.reset_index(), on="Ticker", how="left")
inputs["Asset Type"] = inputs["Asset Type"].fillna(inputs["Ticker"].map(infer_asset_type))
inputs["Sector"] = inputs["Sector"].fillna(inputs["Asset Type"])
inputs["Industry"] = inputs["Industry"].fillna(inputs["Asset Type"])

st.markdown("### Holdings summary")
show_inputs = inputs.copy()
# pretty percent
show_inputs["P/L %"] = show_inputs["P/L %"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
st.dataframe(show_inputs, use_container_width=True)

# Allocation pie
if show_pie:
    st.markdown("## Allocation (by Exposure)")
    pie_df = inputs[["Ticker", "Exposure"]].copy()
    pie_df = pie_df[pie_df["Exposure"] > 0]
    if pie_df.empty:
        st.info("No exposure to plot.")
    else:
        fig_alloc = px.pie(
            pie_df, names="Ticker", values="Exposure", hole=0.45,
            title="Exposure Allocation", template="plotly_white",
        )
        fig_alloc.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_alloc, use_container_width=True)

# =========================================================
# HISTORICAL PRICES
# =========================================================
with st.spinner("Downloading historical prices..."):
    prices = download_prices(inputs["Ticker"].tolist(), start_date, end_date)

if prices.empty:
    st.error("No historical data.")
    st.stop()

# =========================================================
# PER-TICKER PERFORMANCE (P/L-BASED)
# =========================================================
st.markdown("## Portfolio Performance (Per Stock)")

equity = pd.DataFrame(index=prices.index)

for _, r in inputs.iterrows():
    t = r["Ticker"]
    entry = float(r["Entry Price"])
    shares = float(r["Shares"])
    mult = float(r["Multiplier"])
    exposure = float(r["Exposure"])
    pos = str(r["Position"]).strip().lower()

    if pos == "long":
        pl = (prices[t] - entry) * shares * mult
    else:
        pl = (entry - prices[t]) * shares * mult

    equity[t] = 100.0 * (1.0 + (pl / exposure))

perf_long = equity.reset_index().rename(columns={"index": "Date"}).melt(
    id_vars="Date",
    var_name="Ticker",
    value_name="Performance (start=100)",
)

fig_each = px.line(
    perf_long,
    x="Date",
    y="Performance (start=100)",
    color="Ticker",
    title="Per-Stock Performance (P/L-based, Start = 100)",
    template="plotly_white",
)
fig_each.update_layout(height=600, margin=dict(l=20, r=20, t=60, b=40))
fig_each.update_xaxes(rangeslider=dict(visible=True))
st.plotly_chart(fig_each, use_container_width=True)

# =========================================================
# PORTFOLIO LINE (CORRECT)
# =========================================================
st.markdown("## Portfolio Performance")

weights = inputs["Exposure"].astype(float).values
weights = weights / weights.sum() if weights.sum() != 0 else np.ones_like(weights) / len(weights)

returns = (equity.values / 100.0 - 1.0)
port_ret = (returns * weights[None, :]).sum(axis=1)

port_index = 100.0 * (1.0 + port_ret)
port_index = 100.0 * (port_index / port_index[0])

port_df = pd.DataFrame({"Date": prices.index, "Portfolio (start=100)": port_index})

fig_port = px.line(
    port_df, x="Date", y="Portfolio (start=100)",
    title="Portfolio Performance (Start = 100)",
    template="plotly_white",
)
fig_port.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=40))
fig_port.update_xaxes(rangeslider=dict(visible=True))
st.plotly_chart(fig_port, use_container_width=True)

# Drawdown
st.markdown("### Drawdown (Portfolio)")
eq = pd.Series(port_index, index=prices.index)
dd = eq / eq.cummax() - 1.0
dd_df = pd.DataFrame({"Date": dd.index, "Drawdown": dd.values})

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=dd_df["Date"], y=dd_df["Drawdown"], mode="lines", name="Drawdown"))
fig_dd.update_layout(template="plotly_white", height=260, margin=dict(l=20, r=20, t=10, b=40))
fig_dd.update_yaxes(tickformat=".0%")
fig_dd.update_xaxes(rangeslider=dict(visible=True))
st.plotly_chart(fig_dd, use_container_width=True)

# =========================================================
# INDUSTRY / ASSET-TYPE ABUNDANCE (NO UNKNOWN)
# =========================================================
st.markdown("## Industry / Asset-Type Abundance")

group_by = st.radio("Group by", ["Industry", "Sector", "Asset Type"], index=0, horizontal=True)
grp = inputs.groupby(group_by)["Exposure"].sum().sort_values(ascending=False).reset_index()

fig_ind = px.pie(
    grp, names=group_by, values="Exposure", hole=0.4,
    title=f"{group_by} Allocation (by Exposure)", template="plotly_white"
)
fig_ind.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig_ind, use_container_width=True)
st.dataframe(grp, use_container_width=True)

# =========================================================
# NEWS BY STOCK (Yahoo -> Google fallback)
# =========================================================
st.markdown("## Stock-Specific News")

selected = st.multiselect(
    "Select tickers for news",
    inputs["Ticker"].tolist(),
    default=inputs["Ticker"].tolist()[: min(5, len(inputs))],
)
max_items = st.slider("Max headlines per ticker", 3, 25, 10)

for t in selected:
    st.subheader(t)

    df_news = yahoo_news(t, n=max_items)
    if df_news.empty or df_news["title"].fillna("").str.strip().eq("").all():
        st.caption("Source: Google News (RSS)")
        df_news = google_news(t, n=max_items)
    else:
        st.caption("Source: Yahoo Finance (yfinance)")

    render_news(df_news, max_items=max_items)

if show_debug:
    st.markdown("### Debug")
    st.write("Equity tail:")
    st.dataframe(equity.tail(10), use_container_width=True)
