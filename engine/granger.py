# engine/granger.py
# Granger causality tests: does RAI divergence Granger-cause half-life changes?

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from config import PAIRS, GRANGER_MAX_LAG


def run_granger_tests(pair_metrics: dict, pair_rai: dict, pairs_list: list = None) -> pd.DataFrame:
    """
    For each pair, test: does rai_divergence Granger-cause half_life?

    grangercausalitytests(data, maxlag) expects a 2-column array where
    column 0 = the *caused* variable (Y)  and  column 1 = the *causing* variable (X).
    We test: RAI divergence → half-life, so column order is [half_life, rai_divergence].

    Returns a summary DataFrame with F-stats and p-values per pair and lag.
    """
    if pairs_list is None:
        pairs_list = PAIRS

    rows = []
    for pair in pairs_list:
        if pair not in pair_metrics or pair not in pair_rai:
            continue
        metrics = pair_metrics[pair][["half_life"]].dropna()
        rai     = pair_rai[pair][["rai_divergence"]].dropna()
        combined = metrics.join(rai, how="inner").dropna()

        if len(combined) < 10:
            print(f"[granger] {pair}: not enough data ({len(combined)} rows), skipping")
            continue

        # Column order: [Y=half_life, X=rai_divergence]
        data = combined[["half_life", "rai_divergence"]].values
        try:
            test_result = grangercausalitytests(data, maxlag=GRANGER_MAX_LAG, verbose=False)
            for lag in range(1, GRANGER_MAX_LAG + 1):
                f_stat = test_result[lag][0]["ssr_ftest"][0]
                p_val  = test_result[lag][0]["ssr_ftest"][1]
                rows.append({
                    "pair": f"{pair[0]}/{pair[1]}",
                    "lag": lag,
                    "f_stat": round(f_stat, 4),
                    "p_value": round(p_val, 4),
                    "significant_5pct": p_val < 0.05,
                    "significant_10pct": p_val < 0.10,
                })
        except Exception as e:
            print(f"[granger] {pair}: error — {e}")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    pair_metrics = {}
    pair_rai     = {}
    for pair in PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        try:
            pair_metrics[pair] = pd.read_csv(f"results/pair_metrics_{key}.csv", index_col=0, parse_dates=True)
            pair_rai[pair]     = pd.read_csv(f"results/pair_rai_{key}.csv",     index_col=0, parse_dates=True)
        except FileNotFoundError:
            pass
    results = run_granger_tests(pair_metrics, pair_rai)
    results.to_csv("results/granger_results.csv", index=False)
    print(results.to_string())
