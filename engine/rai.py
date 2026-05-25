# engine/rai.py
# Build the RAI divergence signal per pair — the exogenous variable for Granger tests.

import pandas as pd
import numpy as np
from config import PAIRS


def build_pair_rai(rai_norm: pd.DataFrame, pairs_list: list = None) -> dict:
    """
    For each pair (A, B), compute:
      - rai_a, rai_b: individual normalised RAI
      - rai_divergence: |RAI_A - RAI_B|  (the key signal)
      - rai_diff: RAI_A - RAI_B           (signed version)
    Returns dict of DataFrames keyed by pair tuple.
    """
    if pairs_list is None:
        pairs_list = PAIRS

    result = {}
    for pair in pairs_list:
        a, b = pair
        if a not in rai_norm.columns or b not in rai_norm.columns:
            print(f"[rai] skipping pair {pair}: missing RAI data")
            continue
        df = pd.DataFrame({
            "rai_a": rai_norm[a],
            "rai_b": rai_norm[b],
        })
        df["rai_divergence"] = (df["rai_a"] - df["rai_b"]).abs()
        df["rai_diff"]       = df["rai_a"] - df["rai_b"]
        result[pair] = df
    return result


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    rai_norm = pd.read_csv("results/rai_normalised.csv", index_col=0, parse_dates=True)
    pair_rai = build_pair_rai(rai_norm)
    for pair, df in pair_rai.items():
        key = f"{pair[0]}_{pair[1]}"
        df.to_csv(f"results/pair_rai_{key}.csv")
        print(f"Saved pair_rai_{key}.csv")
