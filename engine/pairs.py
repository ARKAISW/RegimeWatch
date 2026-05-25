# engine/pairs.py
# Rolling Johansen cointegration test + OU half-life for each pair over time.

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from config import PAIRS, ROLLING_WINDOW


def johansen_rank(series_a: np.ndarray, series_b: np.ndarray) -> tuple:
    """
    Run Johansen test on a 2-asset system.
    Returns (rank, eigenvalue_0) — rank=1 means cointegrated.
    """
    data = np.column_stack([series_a, series_b])
    try:
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        # trace statistic vs 95% critical value
        rank = int(result.lr1[0] > result.cvt[0, 1])
        eigenvalue = result.eig[0]
        return rank, eigenvalue
    except Exception:
        return 0, np.nan


def ou_half_life(spread: np.ndarray) -> float:
    """
    Estimate OU half-life from spread via OLS regression on lag-1 differences.
    half_life = -log(2) / theta
    """
    spread = spread[~np.isnan(spread)]
    if len(spread) < 20:
        return np.nan
    lag = spread[:-1]
    delta = np.diff(spread)
    X = add_constant(lag)
    try:
        model = OLS(delta, X).fit()
        theta = model.params[1]
        if theta >= 0:
            return np.nan  # not mean-reverting
        return -np.log(2) / theta
    except Exception:
        return np.nan


def compute_hedge_ratio(series_a: np.ndarray, series_b: np.ndarray) -> float:
    """OLS hedge ratio: series_a = beta * series_b + epsilon"""
    X = add_constant(series_b)
    try:
        model = OLS(series_a, X).fit()
        return model.params[1]
    except Exception:
        return np.nan


def compute_rolling_pair_metrics(prices: pd.DataFrame, pairs_list: list = None) -> dict:
    """
    For each pair, compute rolling Johansen rank, eigenvalue, and OU half-life.
    Returns dict of DataFrames keyed by pair tuple.
    """
    if pairs_list is None:
        pairs_list = PAIRS

    results = {}
    for pair in pairs_list:
        a, b = pair
        if a not in prices.columns or b not in prices.columns:
            print(f"[pairs] skipping {pair}: missing price data")
            continue
        pa = prices[a].values
        pb = prices[b].values
        dates = prices.index
        n = len(pa)

        rows = []
        for i in range(ROLLING_WINDOW, n):
            window_a = pa[i - ROLLING_WINDOW:i]
            window_b = pb[i - ROLLING_WINDOW:i]
            rank, eigen = johansen_rank(window_a, window_b)
            beta = compute_hedge_ratio(window_a, window_b)
            spread = window_a - beta * window_b
            hl = ou_half_life(spread)
            # z-score of spread
            zscore = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-9)
            rows.append({
                "date": dates[i],
                "johansen_rank": rank,
                "eigenvalue": eigen,
                "hedge_ratio": beta,
                "half_life": hl,
                "zscore": zscore,
            })

        df = pd.DataFrame(rows).set_index("date")
        results[pair] = df
        print(f"[pairs] computed metrics for {a}/{b}: {len(df)} rows")
    return results


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    prices = pd.read_csv("results/prices.csv", index_col=0, parse_dates=True)
    metrics = compute_rolling_pair_metrics(prices)
    for pair, df in metrics.items():
        key = f"{pair[0]}_{pair[1]}"
        df.to_csv(f"results/pair_metrics_{key}.csv")
        print(f"Saved pair_metrics_{key}.csv")
