# config.py
# Central configuration for RegimeWatch pipeline.

import os
from dotenv import load_dotenv

load_dotenv()

# ── Bright Data credentials ──────────────────────────────────────────
# Dynamic helpers — always return the live value from st.secrets or env
def get_api_key():
    """Resolve API key: Streamlit secrets first, then OS environment."""
    try:
        import streamlit as st
        if "BRIGHTDATA_API_KEY" in st.secrets:
            return st.secrets["BRIGHTDATA_API_KEY"]
        if "brightdata_api_key" in st.secrets:
            return st.secrets["brightdata_api_key"]
    except Exception:
        pass
    return os.getenv("BRIGHTDATA_API_KEY")

def get_serp_zone():
    """Resolve SERP zone: Streamlit secrets first, then OS environment."""
    try:
        import streamlit as st
        if "BRIGHTDATA_SERP_ZONE" in st.secrets:
            return st.secrets["BRIGHTDATA_SERP_ZONE"]
        if "brightdata_serp_zone" in st.secrets:
            return st.secrets["brightdata_serp_zone"]
    except Exception:
        pass
    return os.getenv("BRIGHTDATA_SERP_ZONE", "serp_api1")

# Static fallback for code that does `from config import BRIGHTDATA_API_KEY`
BRIGHTDATA_API_KEY  = get_api_key()
BRIGHTDATA_SERP_ZONE = get_serp_zone()

# ── Pairs to monitor ─────────────────────────────────────────────────
PAIRS = [
    ("JPM", "BAC"),    # financials
    ("XOM", "CVX"),    # energy
    ("JNJ", "PFE"),    # pharma
    ("V", "MA"),       # payments
    ("GS", "MS"),      # investment banks
    ("WFC", "USB"),    # regional banks
    ("MRK", "ABBV"),   # large pharma
    ("SLB", "HAL"),    # oilfield services
    ("BLK", "TROW"),   # asset managers
    ("AXP", "COF"),    # consumer credit
]

# ── Ticker → full company name (for SERP queries) ────────────────────
TICKER_TO_NAME = {
    "JPM":  "JPMorgan Chase",
    "BAC":  "Bank of America",
    "XOM":  "Exxon Mobil",
    "CVX":  "Chevron",
    "JNJ":  "Johnson & Johnson",
    "PFE":  "Pfizer",
    "V":    "Visa",
    "MA":   "Mastercard",
    "GS":   "Goldman Sachs",
    "MS":   "Morgan Stanley",
    "WFC":  "Wells Fargo",
    "USB":  "US Bancorp",
    "MRK":  "Merck",
    "ABBV": "AbbVie",
    "SLB":  "Schlumberger",
    "HAL":  "Halliburton",
    "BLK":  "BlackRock",
    "TROW": "T. Rowe Price",
    "AXP":  "American Express",
    "COF":  "Capital One Financial",
}

# ── Price data window ─────────────────────────────────────────────────
START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

# ── Rolling window for Johansen test and OU half-life ────────────────
ROLLING_WINDOW = 120  # trading days

# ── Granger test max lags ─────────────────────────────────────────────
GRANGER_MAX_LAG = 5

# ── RAI: SERP results per query ──────────────────────────────────────
SERP_NUM_RESULTS = 10

# ── Risk flag thresholds (tune after seeing your data) ───────────────
WATCH_THRESHOLD    = 0.8   # RAI divergence z-score
ELEVATED_THRESHOLD = 1.5
SUSPEND_THRESHOLD  = 2.2
