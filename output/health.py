# output/health.py
# Pair Health Object builder + risk flag logic.

import pandas as pd
import numpy as np
from output.schema import PairHealthObject
from config import PAIRS, WATCH_THRESHOLD, ELEVATED_THRESHOLD, SUSPEND_THRESHOLD


def trend_slope(series: pd.Series, window: int = 7) -> float:
    """Linear slope of last `window` observations."""
    s = series.dropna().tail(window)
    if len(s) < 3:
        return np.nan
    x = np.arange(len(s))
    return float(np.polyfit(x, s.values, 1)[0])


def assign_risk_flag(rai_div: float, hl_trend: float, eigen_trend: float) -> str:
    """
    Assign a risk flag based on RAI divergence level and statistical trend signals.
    """
    if rai_div > SUSPEND_THRESHOLD:
        return "SUSPEND"
    if rai_div > ELEVATED_THRESHOLD or (hl_trend > 2.0 and eigen_trend < -0.05):
        return "ELEVATED"
    if rai_div > WATCH_THRESHOLD:
        return "WATCH"
    return "NORMAL"


def _precompute_rolling_slopes(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Pre-compute rolling linear slope for a series.
    More efficient than computing expanding-then-tail for each row.
    """
    def _slope(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) < 3:
            return np.nan
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]

    return series.rolling(window=window, min_periods=3).apply(_slope, raw=True)


def build_health_objects(
    pair_metrics: dict,
    pair_rai: dict,
    granger_results: pd.DataFrame,
    pairs_list: list = None,
) -> list:
    """
    Build a PairHealthObject for every pair on every date.

    NOTE: `signal_lag` is a pair-level static property derived from the Granger
    results table — it does not vary over time within a pair.
    """
    if pairs_list is None:
        pairs_list = PAIRS

    objects = []
    for pair in pairs_list:
        if pair not in pair_metrics or pair not in pair_rai:
            continue
        metrics = pair_metrics[pair]
        rai     = pair_rai[pair]

        # Ensure datetime index for safe .date() calls
        metrics.index = pd.to_datetime(metrics.index)
        rai.index     = pd.to_datetime(rai.index)

        combined = metrics.join(rai, how="inner").dropna()
        if combined.empty:
            continue

        # Pre-compute rolling slopes (fixes O(n²) expanding-window issue)
        hl_slopes    = _precompute_rolling_slopes(combined["half_life"])
        eigen_slopes = _precompute_rolling_slopes(combined["eigenvalue"])

        # Best Granger lag for this pair (pair-level, not time-varying)
        pair_key = f"{pair[0]}/{pair[1]}"
        signal_lag = -1
        if not granger_results.empty and "pair" in granger_results.columns:
            gr = granger_results[
                (granger_results["pair"] == pair_key) &
                (granger_results["significant_10pct"] == True)
            ]
            signal_lag = int(gr["lag"].min()) if not gr.empty else -1

        for date, row in combined.iterrows():
            hl_trend    = hl_slopes.get(date, np.nan)
            eigen_trend = eigen_slopes.get(date, np.nan)

            flag = assign_risk_flag(
                rai_div=row["rai_divergence"],
                hl_trend=hl_trend if not np.isnan(hl_trend) else 0,
                eigen_trend=eigen_trend if not np.isnan(eigen_trend) else 0,
            )

            obj = PairHealthObject(
                pair=pair,
                date=str(date.date()),
                rai_a=round(row["rai_a"], 4),
                rai_b=round(row["rai_b"], 4),
                rai_divergence=round(row["rai_divergence"], 4),
                johansen_rank=int(row["johansen_rank"]),
                eigenvalue=round(row["eigenvalue"], 4),
                eigenvalue_trend=round(eigen_trend, 6) if not np.isnan(eigen_trend) else None,
                half_life=round(row["half_life"], 2),
                half_life_trend=round(hl_trend, 4) if not np.isnan(hl_trend) else None,
                zscore=round(row["zscore"], 4),
                risk_flag=flag,
                signal_lag=signal_lag,
            )
            objects.append(obj)
    return objects
