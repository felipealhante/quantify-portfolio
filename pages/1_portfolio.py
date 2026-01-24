# ---------------- UI ----------------
tickers_raw = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, NVDA", height=80)
tickers = normalize_tickers(tickers_raw)

today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 2)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Performance start date", value=default_start)
with c2:
    end_date = st.date_input("Performance end date", value=today)

st.markdown("### Trade inputs")
st.write(
    "Enter **Position** (Long/Short), **Entry Price** (Buy for Long, Short-sell for Short), and **Shares**. "
    "If Shares is blank/0, it will default to 1."
)

if not tickers:
    st.info("Add at least one ticker.")
    st.stop()

# Editable table
base_df = pd.DataFrame(
    {
        "Ticker": tickers,
        "Position": ["Long"] * len(tickers),         # NEW
        "Entry Price": [np.nan] * len(tickers),      # NEW (replaces Buy Price)
        "Exit Price (optional)": [np.nan] * len(tickers),  # NEW (optional)
        "Shares": [1.0] * len(tickers),
    }
)

edited = st.data_editor(
    base_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Position": st.column_config.SelectboxColumn(
            options=["Long", "Short"],
            help="Long = buy first. Short = sell first (profit if price falls).",
        ),
        "Entry Price": st.column_config.NumberColumn(
            min_value=0.0, step=0.01,
            help="Long: Buy price. Short: Short-sell price."
        ),
        "Exit Price (optional)": st.column_config.NumberColumn(
            min_value=0.0, step=0.01,
            help="Optional. Long: Sell price. Short: Cover price."
        ),
        "Shares": st.column_config.NumberColumn(min_value=0.0, step=1.0),
    },
)

run = st.button("Run", type="primary", use_container_width=True)
if not run:
    st.stop()

# Validate inputs
inputs = edited.copy()
inputs["Ticker"] = inputs["Ticker"].astype(str).str.upper().str.strip()

# Shares default to 1 if missing/0
inputs["Shares"] = pd.to_numeric(inputs["Shares"], errors="coerce").fillna(1.0)
inputs.loc[inputs["Shares"] <= 0, "Shares"] = 1.0

# Entry Price required
inputs["Entry Price"] = pd.to_numeric(inputs["Entry Price"], errors="coerce")
if inputs["Entry Price"].isna().any():
    st.error("Please fill in Entry Price for all tickers.")
    st.stop()

# Exit optional
inputs["Exit Price (optional)"] = pd.to_numeric(inputs["Exit Price (optional)"], errors="coerce")

# ---------------- Current allocation ----------------
with st.spinner("Fetching latest prices..."):
    latest = get_latest_prices(list(inputs["Ticker"]))

missing = [t for t in inputs["Ticker"] if t not in latest.index]
if missing:
    st.warning(f"Some tickers returned no recent price and will be skipped: {', '.join(missing)}")

inputs = inputs[inputs["Ticker"].isin(latest.index)].copy()
if inputs.empty:
    st.error("No usable tickers after price fetch.")
    st.stop()

inputs["Last Price"] = inputs["Ticker"].map(latest.to_dict()).astype(float)

# Reference price for P/L:
# if user entered Exit, use that; else use Last Price
inputs["Ref Price"] = np.where(
    inputs["Exit Price (optional)"].notna(),
    inputs["Exit Price (optional)"],
    inputs["Last Price"],
).astype(float)

# Exposure for allocation/weights (use absolute so shorts don't create negative weights)
inputs["Exposure"] = (inputs["Entry Price"] * inputs["Shares"]).abs()

# Cost Basis shown as signed notionals (optional):
# Long: +entry*shares, Short: -entry*shares (optional; not used for weights)
inputs["Signed Notional"] = np.where(
    inputs["Position"] == "Long",
    inputs["Entry Price"] * inputs["Shares"],
    -inputs["Entry Price"] * inputs["Shares"],
)

# Current Value as signed mark-to-market (optional display)
inputs["Signed Mkt Value"] = np.where(
    inputs["Position"] == "Long",
    inputs["Last Price"] * inputs["Shares"],
    -inputs["Last Price"] * inputs["Shares"],
)

# P/L in currency
inputs["P/L"] = np.where(
    inputs["Position"] == "Long",
    (inputs["Ref Price"] - inputs["Entry Price"]) * inputs["Shares"],
    (inputs["Entry Price"] - inputs["Ref Price"]) * inputs["Shares"],
)

# P/L % relative to exposure
inputs["P/L %"] = np.where(inputs["Exposure"] > 0, inputs["P/L"] / inputs["Exposure"], np.nan)

total_exposure = float(inputs["Exposure"].sum())
inputs["Weight"] = np.where(total_exposure > 0, inputs["Exposure"] / total_exposure, np.nan)

m1, m2, m3 = st.columns(3)
m1.metric("Total Exposure", f"{total_exposure:,.2f}")
m2.metric("Net Notional", f"{float(inputs['Signed Notional'].sum()):,.2f}")
m3.metric("Total P/L", f"{float(inputs['P/L'].sum()):,.2f}")

st.markdown("### Allocation (by exposure)")
fig_pie = px.pie(
    inputs,
    names="Ticker",
    values="Exposure",
    hole=0.45,
    template="plotly_white",
)
fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("### Holdings summary")
show_df = inputs[
    ["Ticker", "Position", "Shares", "Entry Price", "Last Price", "Exit Price (optional)", "Exposure", "P/L", "P/L %", "Weight"]
].copy()
show_df["P/L %"] = show_df["P/L %"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
show_df["Weight"] = show_df["Weight"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
st.dataframe(show_df, use_container_width=True)
