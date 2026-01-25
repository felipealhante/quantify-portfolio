import datetime as dt
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# =========================
# Page config + dark styling
# =========================
st.set_page_config(page_title="GDP Explorer (World Bank Data)", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background-color:#0b0f14; }
      [data-testid="stSidebar"] { background:#0f172a; }
      h1,h2,h3,p,span,div,label { color:#e6edf3 !important; }
      .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("World Bank GDP Explorer")
st.caption(
    "Interactive: hover labels, zoom/drag, range slider, multi-country (max 6). "
    "Growth bars are always shown underneath. GDP comparison is optional."
)

# =========================
# World Bank API helpers
# =========================
WB_BASE = "https://api.worldbank.org/v2"
GDP_CURRENT_USD = "NY.GDP.MKTP.CD"      # GDP (current US$)
GDP_GROWTH_PCT  = "NY.GDP.MKTP.KD.ZG"   # GDP growth (annual %)


@st.cache_data(show_spinner=False)
def wb_get(url: str, params: dict):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def wb_countries() -> pd.DataFrame:
    js = wb_get(f"{WB_BASE}/country", {"format": "json", "per_page": 400})
    rows = js[1] if isinstance(js, list) and len(js) > 1 else []
    df = pd.json_normalize(rows)
    if df.empty:
        return pd.DataFrame(columns=["code", "country", "region"])
    df = df[df["region.value"] != "Aggregates"].copy()
    df = df[["id", "name", "region.value"]].rename(
        columns={"id": "code", "name": "country", "region.value": "region"}
    )
    return df.sort_values("country").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def wb_series(country_code: str, indicator_code: str) -> pd.DataFrame:
    js = wb_get(
        f"{WB_BASE}/country/{country_code}/indicator/{indicator_code}",
        {"format": "json", "per_page": 20000},
    )
    if not isinstance(js, list) or len(js) < 2 or js[1] is None:
        return pd.DataFrame(columns=["year", "value"])

    df = pd.json_normalize(js[1])[["date", "value"]].rename(columns={"date": "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).sort_values("year")
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


# =========================
# Sidebar inputs
# =========================
countries = wb_countries()
if countries.empty:
    st.error("Could not load the World Bank country list.")
    st.stop()

code_map = countries.set_index("country")["code"].to_dict()

with st.sidebar:
    st.header("Inputs")

    default = [c for c in ["Brazil", "United States", "China"] if c in code_map]
    selected = st.multiselect(
        "Countries (max 6)",
        options=countries["country"].tolist(),
        default=default,
    )

    if len(selected) == 0:
        st.info("Select at least 1 country.")
        st.stop()
    if len(selected) > 6:
        st.error("Max 6 countries.")
        st.stop()

    today_year = dt.date.today().year
    start_year, end_year = st.slider(
        "Year range",
        min_value=1960,
        max_value=today_year,
        value=(1990, today_year),
    )

    show_current_gdp = st.checkbox("Show GDP comparison", value=True)

# =========================
# Fetch data
# =========================
with st.spinner("Fetching World Bank data..."):
    gdp_rows = []
    grw_rows = []
    for c in selected:
        code = code_map[c]

        g = wb_series(code, GDP_CURRENT_USD)
        g["country"] = c
        gdp_rows.append(g)

        gr = wb_series(code, GDP_GROWTH_PCT)
        gr["country"] = c
        grw_rows.append(gr)

gdp = pd.concat(gdp_rows, ignore_index=True) if gdp_rows else pd.DataFrame(columns=["year", "value", "country"])
grw = pd.concat(grw_rows, ignore_index=True) if grw_rows else pd.DataFrame(columns=["year", "value", "country"])

gdp = gdp.rename(columns={"value": "gdp"})
grw = grw.rename(columns={"value": "growth"})

gdp = gdp[(gdp["year"] >= start_year) & (gdp["year"] <= end_year)].copy()
grw = grw[(grw["year"] >= start_year) & (grw["year"] <= end_year)].copy()

# =========================
# Main chart: GDP (line)
# =========================
st.subheader("GDP (current US$)")

df_gdp = gdp.dropna(subset=["gdp"]).copy()
if df_gdp.empty:
    st.error("No GDP data available for the selected range.")
    st.stop()

fig = go.Figure()
for c in selected:
    d = df_gdp[df_gdp["country"] == c].sort_values("year")
    fig.add_trace(
        go.Scatter(
            x=d["year"],
            y=d["gdp"],
            mode="lines+markers",
            name=c,
            customdata=[c] * len(d),
            hovertemplate="<b>%{customdata}</b> (%{x})<br>%{y:.3s}<extra></extra>",
        )
    )

fig.update_layout(
    template="plotly_dark",
    height=540,
    margin=dict(l=20, r=20, t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_yaxes(title="GDP (current US$)", tickformat="~s")
fig.update_xaxes(title="Year", rangeslider=dict(visible=True), type="linear")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

# =========================
# ALWAYS show: GDP growth (bars)
# =========================
st.subheader("Annual GDP growth (bars)")

df2 = grw.dropna(subset=["growth"]).copy()
if df2.empty:
    st.info("No GDP growth data available for the selected range.")
else:
    fig2 = go.Figure()
    for c in selected:
        d = df2[df2["country"] == c].sort_values("year")
        fig2.add_trace(
            go.Bar(
                x=d["year"],
                y=d["growth"],
                name=c,
                customdata=[c] * len(d),
                hovertemplate="<b>%{customdata}</b> (%{x})<br>%{y:.2f}%<extra></extra>",
                opacity=0.85,
            )
        )

    fig2.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=20, r=20, t=10, b=30),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig2.update_yaxes(title="GDP growth (%)", ticksuffix="%")
    fig2.update_xaxes(title="Year")

    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

# =========================
# OPTIONAL: Current GDP comparison (latest available year in the selected range)
# =========================
if show_current_gdp:
    st.subheader("Current GDP (latest available year) — comparison")

    rows = []
    for c in selected:
        d = df_gdp[df_gdp["country"] == c].dropna(subset=["gdp"]).sort_values("year")
        if d.empty:
            continue
        rows.append({"Country": c, "Year": int(d["year"].iloc[-1]), "GDP": float(d["gdp"].iloc[-1])})

    if not rows:
        st.info("No GDP values available to compare for the selected countries.")
    else:
        bar_df = pd.DataFrame(rows).sort_values("GDP", ascending=False)

        fig_curr = go.Figure()
        fig_curr.add_bar(
            x=bar_df["Country"],
            y=bar_df["GDP"],
            customdata=bar_df[["Year"]],
            hovertemplate="<b>%{x}</b><br>Year: %{customdata[0]}<br>GDP: %{y:,.0f}<extra></extra>",
            opacity=0.9,
        )

        fig_curr.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=10, b=40),
        )
        fig_curr.update_yaxes(title="GDP (current US$)", tickformat="~s")
        fig_curr.update_xaxes(title="Country")

        st.plotly_chart(fig_curr, use_container_width=True, config={"displayModeBar": True})

# =========================
# Raw data
# =========================
with st.expander("Show raw data"):
    st.markdown("**GDP (current US$)**")
    st.dataframe(df_gdp, use_container_width=True)
    st.markdown("**GDP growth (annual %)**")
    st.dataframe(df2, use_container_width=True)
