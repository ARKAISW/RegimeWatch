# main.py
# Entry point — runs the full RegimeWatch pipeline end to end.

import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.makedirs("results", exist_ok=True)

from data.fetch_prices import fetch_prices
from data.fetch_rai    import build_rai_series, normalise_rai
from engine.pairs      import compute_rolling_pair_metrics
from engine.rai        import build_pair_rai
from engine.granger    import run_granger_tests
from output.health     import build_health_objects
from config            import PAIRS

print("=" * 60)
print("  RegimeWatch — Cointegration Breakdown Monitor")
print("=" * 60)

# ── 1. Prices ────────────────────────────────────────────────────────
print("\n[1/5] Fetching price data ...")
prices = fetch_prices()
prices.to_csv("results/prices.csv")

# ── 2. RAI via Bright Data ───────────────────────────────────────────
print("\n[2/5] Fetching Regulatory Attention Index (Bright Data) ...")
tickers = sorted({t for pair in PAIRS for t in pair})

# Use the actual trading dates from the price index for alignment (fixes D1).
# Limit to last 20 trading days to keep pipeline fast for demos.
dates = prices.index.tolist()[-20:]

rai_raw  = build_rai_series(tickers, dates)
rai_norm = normalise_rai(rai_raw)
rai_raw.to_csv("results/rai_raw.csv")
rai_norm.to_csv("results/rai_normalised.csv")

# ── 3. Rolling pair metrics ──────────────────────────────────────────
print("\n[3/5] Computing rolling Johansen + OU half-life ...")
pair_metrics = compute_rolling_pair_metrics(prices)
pair_rai     = build_pair_rai(rai_norm)

# Save intermediate pair data for debugging
for pair, df in pair_metrics.items():
    key = f"{pair[0]}_{pair[1]}"
    df.to_csv(f"results/pair_metrics_{key}.csv")
for pair, df in pair_rai.items():
    key = f"{pair[0]}_{pair[1]}"
    df.to_csv(f"results/pair_rai_{key}.csv")

# ── 4. Granger causality ────────────────────────────────────────────
print("\n[4/5] Running Granger causality tests ...")
granger_results = run_granger_tests(pair_metrics, pair_rai)
granger_results.to_csv("results/granger_results.csv", index=False)
print(granger_results.to_string())

# ── 5. Pair Health Objects ───────────────────────────────────────────
print("\n[5/5] Building Pair Health Objects ...")
health_objects = build_health_objects(pair_metrics, pair_rai, granger_results)
records = [o.to_dict() for o in health_objects]
with open("results/pair_health.json", "w") as f:
    json.dump(records, f, indent=2, default=str)

# Latest snapshot per pair
latest = {}
for o in health_objects:
    key = o.to_dict()["pair"]  # now a string "A/B"
    latest[key] = o.to_dict()

print("\n📊 Latest Pair Health Snapshot:")
print("-" * 60)
for pair_key, data in latest.items():
    flag = data["risk_flag"]
    icon = {"NORMAL": "✅", "WATCH": "👀", "ELEVATED": "⚠️", "SUSPEND": "🚨"}.get(flag, "?")
    print(f"{icon} {pair_key:<12} | flag={flag:<9} | hl={data['half_life']:.1f}d | rai_div={data['rai_divergence']:.2f} | lag={data['signal_lag']}")

print("\n✅ Done. Results in results/")
