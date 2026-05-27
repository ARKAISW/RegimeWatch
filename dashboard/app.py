# dashboard/app.py
# Self-contained Streamlit dashboard for RegimeWatch.
# Run with:  streamlit run regimewatch/dashboard/app.py
#         or cd regimewatch && streamlit run dashboard/app.py

import sys
import os

# ── Resolve project root so imports and file paths work from any CWD ─
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Now safe to import project modules ───────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Resolve logo path
LOGO_PATH = os.path.join(PROJECT_ROOT, "assets", "logo.png")

st.set_page_config(page_title="RegimeWatch", layout="wide", page_icon=LOGO_PATH)

# ── Helper: resolve path inside results/ ────────────────────────────
def rpath(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


# ── Helper: load pre-computed health data ────────────────────────────
def load_health_data(use_custom: bool = False):
    filename = "pair_health_custom.json" if use_custom else "pair_health.json"
    path = rpath(filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            records = json.load(f)
        if not records:
            return None
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["pair_label"] = df["pair"]
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
st.sidebar.markdown(
    """
    ### 🔍 RegimeWatch
    Monitors cointegrated equity pairs for
    regulatory-attention-driven breakdown using
    **Bright Data's SERP API**.

    ---
    🏗 Built for **Web Data UNLOCKED Hackathon**
    *lablab.ai × Bright Data*

    ---
    """
)

# ── Analysis Mode selection ──────────────────────────────────────────
analysis_mode = st.sidebar.radio(
    "Choose Analysis Mode",
    ["📈 Preset Pairs", "🧪 Custom Ticker Pair Analyzer"],
    help="Preset Pairs loads pre-computed market vectors. Custom Tickers lets you analyze any assets on-the-fly."
)

use_custom_mode = (analysis_mode == "🧪 Custom Ticker Pair Analyzer")

# ── Custom Ticker Pair Analyzer controls ─────────────────────────────
if use_custom_mode:
    st.sidebar.markdown("#### ⚙️ Custom Pair Configuration")
    custom_a = st.sidebar.text_input("Ticker A (e.g. KO)", "KO").strip().upper()
    custom_b = st.sidebar.text_input("Ticker B (e.g. PEP)", "PEP").strip().upper()
    
    run_custom = st.sidebar.button("▶️ Analyze Custom Pair", use_container_width=True)

    if run_custom:
        if not custom_a or not custom_b:
            st.sidebar.error("Please enter both tickers.")
        elif custom_a == custom_b:
            st.sidebar.error("Tickers must be different.")
        else:
            # Load .env for API keys
            from dotenv import load_dotenv
            load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

            import config
            from data.fetch_prices import fetch_prices
            from data.fetch_rai import build_rai_series, normalise_rai
            from engine.pairs import compute_rolling_pair_metrics
            from engine.rai import build_pair_rai
            from engine.granger import run_granger_tests
            from output.health import build_health_objects

            _api_key = config.get_api_key()
            if not _api_key or _api_key == "your_api_key_here":
                keys_found = list(st.secrets.keys()) if hasattr(st, "secrets") else []
                st.sidebar.error(
                    "⚠️ Bright Data API key not set.\n\n"
                    f"**Secrets found in Streamlit Cloud:** `{keys_found}`\n\n"
                    "Please ensure it is set as `BRIGHTDATA_API_KEY = \"your_key\"` in your Streamlit Cloud Secrets Settings."
                )
            else:
                status = st.status(f"🚀 Analyzing {custom_a}/{custom_b} on-the-fly...", expanded=True)
                try:
                    # Step 1: Fetch Prices
                    status.write("📈 **[1/5]** Fetching custom prices from Yahoo Finance...")
                    custom_prices = fetch_prices([custom_a, custom_b])
                    status.write(f"   ✅ Fetched {len(custom_prices)} trading days for {custom_a} and {custom_b}")

                    # Step 2: Fetch RAI
                    status.write(f"🔎 **[2/5]** Fetching live regulatory indexes from Bright Data...")
                    custom_dates = custom_prices.index.tolist()[-20:]
                    
                    # Ensure dynamic name map exists
                    from config import TICKER_TO_NAME
                    TICKER_TO_NAME[custom_a] = TICKER_TO_NAME.get(custom_a, custom_a)
                    TICKER_TO_NAME[custom_b] = TICKER_TO_NAME.get(custom_b, custom_b)

                    # Only 40 API calls total, parallel fetch takes ~10-15s!
                    status.write(f"   → 2 tickers × 20 dates = 40 parallel calls")
                    custom_rai_raw = build_rai_series([custom_a, custom_b], custom_dates, max_workers=8)
                    if custom_rai_raw.sum().sum() == 0:
                        st.sidebar.error("⚠️ **API Fetch Failed:** Bright Data returned 0 for all queries. Please check the Streamlit Cloud logs (Manage app > Logs) to see if your API key is invalid, rate-limited, or out of credits.")
                        status.write("   ⚠️ WARNING: All regulatory signals returned 0 (API failure).")
                    
                    custom_rai_norm = normalise_rai(custom_rai_raw)
                    status.write("   ✅ Regulatory signals normalized")

                    # Step 3: Rolling pair metrics
                    status.write("📊 **[3/5]** Computing rolling Johansen cointegration rank & half-life...")
                    custom_pair = (custom_a, custom_b)
                    custom_pair_metrics = compute_rolling_pair_metrics(custom_prices, [custom_pair])
                    custom_pair_rai = build_pair_rai(custom_rai_norm, [custom_pair])
                    status.write("   ✅ Cointegration stability vectors generated")

                    # Step 4: Granger causality
                    status.write("🧪 **[4/5]** Performing Granger causality hypothesis tests...")
                    custom_granger = run_granger_tests(custom_pair_metrics, custom_pair_rai, [custom_pair])
                    status.write("   ✅ Granger statistics computed")

                    # Step 5: Pair Health scoring
                    status.write("🏥 **[5/5]** Packaging Pair Health Objects...")
                    custom_health = build_health_objects(custom_pair_metrics, custom_pair_rai, custom_granger, [custom_pair])
                    custom_records = [o.to_dict() for o in custom_health]

                    # Save to custom outputs
                    with open(rpath("pair_health_custom.json"), "w") as f:
                        json.dump(custom_records, f, indent=2, default=str)
                    
                    if not custom_granger.empty:
                        custom_granger.to_csv(rpath("granger_results_custom.csv"), index=False)
                    else:
                        # Empty placeholder so load won't fail
                        pd.DataFrame().to_csv(rpath("granger_results_custom.csv"), index=False)

                    status.update(label=f"✅ Analysis for {custom_a}/{custom_b} completed!", state="complete")
                    st.session_state["has_custom_run"] = True
                    st.rerun()

                except Exception as e:
                    status.update(label="❌ Analysis failed", state="error")
                    st.error(f"Error during custom run: {e}")

else:
    # ── Preset Pipeline runner in sidebar ───────────────────────────────────────
    with st.sidebar.expander("🔄 Run / Refresh Preset Pipeline", expanded=False):
        st.markdown(
            "Fetch live regulatory attention data for all **10 default pairs** from Bright Data's SERP API."
        )
        run_all = st.button("▶️  Run Preset Pipeline", use_container_width=True)

    # ── Run the preset pipeline if button pressed ───────────────────────────────
    if run_all:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

        import config
        from config import PAIRS
        from data.fetch_prices import fetch_prices
        from data.fetch_rai import build_rai_series, normalise_rai
        from engine.pairs import compute_rolling_pair_metrics
        from engine.rai import build_pair_rai
        from engine.granger import run_granger_tests
        from output.health import build_health_objects

        _api_key = config.get_api_key()
        if not _api_key or _api_key == "your_api_key_here":
            keys_found = list(st.secrets.keys()) if hasattr(st, "secrets") else []
            st.sidebar.error(
                "⚠️ Bright Data API key not set.\n\n"
                f"**Secrets found in Streamlit Cloud:** `{keys_found}`\n\n"
                "Please ensure it is set as `BRIGHTDATA_API_KEY = \"your_key\"` in your Streamlit Cloud Secrets Settings."
            )
        else:
            status = st.status("🚀 Running RegimeWatch preset pipeline…", expanded=True)

            # Step 1: Prices
            status.write("📈 **[1/5]** Fetching price data from Yahoo Finance…")
            prices = fetch_prices()
            prices.to_csv(rpath("prices.csv"))
            status.write(f"   ✅ {len(prices)} trading days, {len(prices.columns)} tickers")

            # Step 2: RAI via Bright Data (Concurrent fetch)
            status.write("🔎 **[2/5]** Querying Bright Data SERP API for regulatory news…")
            tickers = sorted({t for pair in PAIRS for t in pair})
            dates = prices.index.tolist()[-20:]
            status.write(f"   → {len(tickers)} tickers × {len(dates)} dates = {len(tickers)*len(dates)} API calls (8 concurrent)")
            status.write("   ⏳ Estimated ~4 minutes total…")
            rai_raw = build_rai_series(tickers, dates, max_workers=8)
            if rai_raw.sum().sum() == 0:
                st.sidebar.error("⚠️ **API Fetch Failed:** Bright Data returned 0 for all queries. Please check the Streamlit Cloud logs (Manage app > Logs) to see if your API key is invalid, rate-limited, or out of credits.")
                status.write("   ⚠️ WARNING: All regulatory signals returned 0 (API failure).")
                
            rai_norm = normalise_rai(rai_raw)
            rai_raw.to_csv(rpath("rai_raw.csv"))
            rai_norm.to_csv(rpath("rai_normalised.csv"))
            status.write("   ✅ RAI data fetched and normalised")

            # Step 3: Rolling pair metrics
            status.write("📊 **[3/5]** Computing rolling Johansen + OU half-life…")
            pair_metrics = compute_rolling_pair_metrics(prices)
            pair_rai = build_pair_rai(rai_norm)
            for pair, df in pair_metrics.items():
                key = f"{pair[0]}_{pair[1]}"
                df.to_csv(rpath(f"pair_metrics_{key}.csv"))
            for pair, df in pair_rai.items():
                key = f"{pair[0]}_{pair[1]}"
                df.to_csv(rpath(f"pair_rai_{key}.csv"))
            status.write("   ✅ Pair metrics computed")

            # Step 4: Granger causality
            status.write("🧪 **[4/5]** Running Granger causality tests…")
            granger_results = run_granger_tests(pair_metrics, pair_rai)
            granger_results.to_csv(rpath("granger_results.csv"), index=False)
            if "significant_10pct" in granger_results.columns:
                sig = granger_results[granger_results["significant_10pct"] == True]
                status.write(f"   ✅ {len(sig)} significant lag-pair combinations found")
            else:
                status.write("   ⚠️ Granger test returned no results (not enough overlapping data)")

            # Step 5: Pair Health Objects
            status.write("🏥 **[5/5]** Building Pair Health Objects…")
            health_objects = build_health_objects(pair_metrics, pair_rai, granger_results)
            records = [o.to_dict() for o in health_objects]
            with open(rpath("pair_health.json"), "w") as f:
                json.dump(records, f, indent=2, default=str)
            status.write(f"   ✅ {len(records)} health records generated")

            status.update(label="✅ Preset Pipeline complete!", state="complete")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════

# Display logo + title
_logo_col, _title_col = st.columns([0.06, 0.94])
with _logo_col:
    st.image(LOGO_PATH, width=64)
with _title_col:
    st.title("RegimeWatch")
    st.caption("Live Regulatory Attention as a Leading Indicator of Cointegration Breakdown")

# Load correct data according to mode
health_df = load_health_data(use_custom=use_custom_mode)

if health_df is None:
    if use_custom_mode:
        st.warning(
            "No custom analysis run yet. Use the sidebar controls to choose two custom tickers "
            "and click **▶️ Analyze Custom Pair** to run the live analysis!"
        )
    else:
        st.warning(
            "No preset results found. Click **🔄 Run / Refresh Preset Pipeline** in the sidebar "
            "to fetch live data from Bright Data and compute preset metrics."
        )
    st.stop()

# ── Pair selector ────────────────────────────────────────────────────
pairs = sorted(health_df["pair_label"].unique().tolist())
selected_pair = st.sidebar.selectbox("Select pair to view", pairs)
df = health_df[health_df["pair_label"] == selected_pair].sort_values("date")

# ── Latest health card ───────────────────────────────────────────────
latest = df.iloc[-1]
flag = latest["risk_flag"]
flag_emoji = {"NORMAL": "✅", "WATCH": "👀", "ELEVATED": "⚠️", "SUSPEND": "🚨"}.get(flag, "❓")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Risk Flag", f"{flag_emoji} {flag}")
col2.metric("OU Half-Life", f"{latest['half_life']:.1f} days")
col3.metric("RAI Divergence", f"{latest['rai_divergence']:.2f}σ")
col4.metric(
    "Granger Signal Lag",
    f"{latest['signal_lag']} days" if latest["signal_lag"] > 0 else "n/s",
)

st.markdown("---")

# ── Chart 1: RAI divergence + half-life (dual axis) ─────────────────
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(
        x=df["date"], y=df["rai_divergence"],
        name="RAI Divergence", line=dict(color="#ef9f27", width=1.5),
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=df["date"], y=df["half_life"],
        name="OU Half-Life (days)", line=dict(color="#378add", width=1.5),
    ),
    secondary_y=True,
)
fig.update_layout(
    title=f"{selected_pair} — RAI Divergence vs OU Half-Life",
    hovermode="x unified", height=380,
    legend=dict(orientation="h", y=-0.15),
)
fig.update_yaxes(title_text="RAI Divergence (σ)", secondary_y=False)
fig.update_yaxes(title_text="OU Half-Life (days)", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# ── Chart 2: Eigenvalue + Johansen rank ──────────────────────────────
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(
    go.Scatter(
        x=df["date"], y=df["eigenvalue"],
        name="Johansen Eigenvalue", line=dict(color="#1d9e75", width=1.5),
    ),
    secondary_y=False,
)
fig2.add_trace(
    go.Scatter(
        x=df["date"], y=df["johansen_rank"],
        name="Cointegration Rank", line=dict(color="#d85a30", width=1, dash="dot"),
    ),
    secondary_y=True,
)
fig2.update_layout(
    title=f"{selected_pair} — Cointegration Stability",
    hovermode="x unified", height=300,
    legend=dict(orientation="h", y=-0.2),
)
fig2.update_yaxes(title_text="Eigenvalue", secondary_y=False)
fig2.update_yaxes(title_text="Rank", range=[-0.1, 2.1], secondary_y=True)
st.plotly_chart(fig2, use_container_width=True)

# ── Granger results table ────────────────────────────────────────────
st.markdown("### Granger Causality Results (RAI → Half-Life)")
gr_path = rpath("granger_results_custom.csv" if use_custom_mode else "granger_results.csv")
if os.path.exists(gr_path):
    try:
        gr = pd.read_csv(gr_path)
        if not gr.empty:
            gr_pair = gr[gr["pair"] == selected_pair]
            if not gr_pair.empty:
                def _highlight_sig(row):
                    """Highlight rows where p < 0.05 in green."""
                    bg = "background-color: #c8f7c5" if row["p_value"] < 0.05 else ""
                    return [bg] * len(row)
                st.dataframe(
                    gr_pair.style.apply(_highlight_sig, axis=1),
                    use_container_width=True,
                )
            else:
                st.info("No Granger results for this pair.")
        else:
            st.info("Granger tests run, but no relationships scored.")
    except Exception:
        st.info("No Granger causality results available yet.")
else:
    st.info("Granger results not yet computed.")

# ── All-pairs health table ───────────────────────────────────────────
st.markdown("### Latest Health Snapshot Table")
latest_all = health_df.sort_values("date").groupby("pair_label").last().reset_index()
display_cols = [
    "pair_label", "risk_flag", "half_life", "rai_divergence",
    "eigenvalue", "johansen_rank", "signal_lag",
]
latest_all = latest_all[display_cols]
latest_all.columns = [
    "Pair", "Risk Flag", "Half-Life (d)", "RAI Divergence",
    "Eigenvalue", "Johansen Rank", "Signal Lag (d)",
]


def flag_color(val):
    colors = {
        "NORMAL":   "background-color: #d1e7dd; color: #0f5132; font-weight: bold;",
        "WATCH":    "background-color: #fff3cd; color: #664d03; font-weight: bold;",
        "ELEVATED": "background-color: #f8d7da; color: #842029; font-weight: bold;",
        "SUSPEND":  "background-color: #f8d7da; color: #842029; font-weight: bold; border: 2px solid #f5c2c7;",
    }
    return colors.get(val, "")


try:
    styled = latest_all.style.map(flag_color, subset=["Risk Flag"])
except AttributeError:
    styled = latest_all.style.applymap(flag_color, subset=["Risk Flag"])

st.dataframe(styled, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "RegimeWatch • Web Data UNLOCKED Hackathon • lablab.ai × Bright Data<br>"
    "Powered by Bright Data SERP API for live regulatory attention signals"
    "</div>",
    unsafe_allow_html=True,
)
