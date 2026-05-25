# data/fetch_prices.py
# Fetch adjusted close prices for all tickers via yfinance.

import os
import yfinance as yf
import pandas as pd
from config import PAIRS, START_DATE, END_DATE


def get_all_tickers() -> list:
    """Extract unique tickers from configured pairs."""
    tickers = set()
    for a, b in PAIRS:
        tickers.add(a)
        tickers.add(b)
    return sorted(tickers)


def fetch_prices(tickers: list = None) -> pd.DataFrame:
    """
    Download adjusted close prices for tickers.
    Returns a DataFrame with DatetimeIndex and one column per ticker.
    """
    if tickers is None:
        tickers = get_all_tickers()
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True)

    # yfinance returns MultiIndex columns when >1 ticker: (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker edge-case: raw is already a flat DataFrame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Forward-fill gaps (market holidays within bdate_range), then drop
    # only rows where *all* tickers are NaN.
    prices = prices.ffill().dropna(how="all")
    print(f"[prices] fetched {len(prices)} rows, {len(prices.columns)} tickers")
    return prices


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    df = fetch_prices()
    df.to_csv("results/prices.csv")
    print(df.tail())
