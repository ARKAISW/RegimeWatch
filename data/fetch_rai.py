# data/fetch_rai.py
# Fetch Regulatory Attention Index via Bright Data SERP API.
#
# Bright Data SERP API contract (from docs):
#   POST https://api.brightdata.com/request
#   Headers:  Authorization: Bearer <API_KEY>
#             Content-Type: application/json
#   Body:     { "zone": "<ZONE>", "url": "<google_search_url>", "format": "raw",
#               "data_format": "parsed_light" }
#   Response: { "organic": [ { "title", "description", "link", "global_rank" }, ... ] }

import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import config
from config import PAIRS, TICKER_TO_NAME

# Keep local env loading active
load_dotenv()

SERP_ENDPOINT = "https://api.brightdata.com/request"

REGULATORY_KEYWORDS = "regulatory OR SEC OR lawsuit OR probe OR investigation OR fine OR penalty OR compliance"

# Keywords used for naive negative-sentiment scoring
NEGATIVE_WORDS = [
    "probe", "sue", "fine", "penalty", "fraud",
    "violation", "investigation", "lawsuit", "indictment",
    "enforcement", "sanction", "subpoena",
]


def _build_google_url(query: str, num: int = 10, date_str: str = None) -> str:
    """
    Build a Google search URL with optional date-scoping via the `tbs` parameter.
    `date_str` should be in MM/DD/YYYY format.
    """
    import urllib.parse
    params = {
        "q": query,
        "num": str(num),
        "gl": "us",
        "hl": "en",
    }
    if date_str:
        params["tbs"] = f"cdr:1,cd_min:{date_str},cd_max:{date_str}"
    return "https://www.google.com/search?" + urllib.parse.urlencode(params)


def query_rai(ticker: str, date_str: str) -> dict:
    """
    Query Bright Data SERP API for regulatory news about a company on a given date.
    Uses the full company name (via TICKER_TO_NAME) for better search relevance.
    Returns { count, neg_score, raw }.
    """
    company_name = TICKER_TO_NAME.get(ticker, ticker)
    query = f'"{company_name}" {REGULATORY_KEYWORDS}'
    google_url = _build_google_url(query, num=10, date_str=date_str)

    api_key   = config.get_api_key()
    serp_zone = config.get_serp_zone()

    payload = {
        "zone": serp_zone,
        "url": google_url,
        "format": "raw",
        "data_format": "parsed_light",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            SERP_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # parsed_light format returns { "organic": [ ... ] }
        results = data.get("organic", [])
        count = len(results)

        # Naive negative-sentiment scoring
        neg_score = 0
        for r in results:
            text = (r.get("title", "") + " " + r.get("description", "")).lower()
            neg_score += sum(text.count(w) for w in NEGATIVE_WORDS)

        return {"count": count, "neg_score": neg_score, "raw": count + neg_score * 0.5}

    except requests.exceptions.HTTPError as e:
        print(f"[rai] HTTP error for {ticker} on {date_str}: {e} — {resp.text[:200]}")
        return {"count": 0, "neg_score": 0, "raw": 0}
    except Exception as e:
        print(f"[rai] error for {ticker} on {date_str}: {e}")
        return {"count": 0, "neg_score": 0, "raw": 0}


def build_rai_series(tickers: list, dates: list, max_workers: int = 8) -> pd.DataFrame:
    """
    For each ticker and date, fetch RAI — using concurrent requests.
    Bright Data's SERP API is designed for high-throughput scraping,
    so parallel requests are safe and dramatically faster.
    Returns DataFrame: index=date, columns=tickers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Resolve results dir relative to project root (where config.py lives)
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _results_dir = os.path.join(_project_root, "results")
    os.makedirs(_results_dir, exist_ok=True)

    # Build flat list of all (ticker, date) tasks
    tasks = [(ticker, date) for ticker in tickers for date in dates]
    total = len(tasks)
    results = {}  # (ticker, date) -> raw score
    done_count = 0

    print(f"[rai] launching {total} queries across {max_workers} threads ...")

    def _fetch_one(ticker, date):
        date_str = date.strftime("%m/%d/%Y")
        return query_rai(ticker, date_str)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ticker, date in tasks:
            fut = executor.submit(_fetch_one, ticker, date)
            futures[fut] = (ticker, date)

        for fut in as_completed(futures):
            ticker, date = futures[fut]
            try:
                result = fut.result()
                results[(ticker, date)] = result["raw"]
            except Exception as e:
                print(f"[rai] error {ticker} {date}: {e}")
                results[(ticker, date)] = 0
            done_count += 1
            if done_count % 20 == 0 or done_count == total:
                print(f"[rai]   progress: {done_count}/{total}")

    # Reassemble into DataFrame and cache per ticker
    records = {}
    for ticker in tickers:
        series = [results.get((ticker, d), 0) for d in dates]
        records[ticker] = series
        cache_path = os.path.join(_results_dir, f"rai_cache_{ticker}.csv")
        pd.DataFrame({"date": dates, ticker: series}).set_index("date").to_csv(cache_path)
        print(f"[rai]   → cached {ticker} ({len(series)} days)")

    df = pd.DataFrame(records, index=dates)
    return df


def normalise_rai(rai_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalise each ticker's RAI across time.
    NOTE: Uses full-sample mean/std — introduces look-ahead bias.
          Acceptable for hackathon; production should use expanding window.
    """
    return (rai_raw - rai_raw.mean()) / (rai_raw.std() + 1e-9)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    tickers = sorted({t for pair in PAIRS for t in pair})
    # Use last 60 trading days to stay within rate limits and budget
    from config import START_DATE, END_DATE
    dates = pd.bdate_range(START_DATE, END_DATE).tolist()[-60:]
    raw = build_rai_series(tickers, dates)
    raw.to_csv("results/rai_raw.csv")
    norm = normalise_rai(raw)
    norm.to_csv("results/rai_normalised.csv")
    print(norm.tail())
