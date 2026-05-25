# RegimeWatch 🔍
### *Detecting Cointegration Breakdown in Pairs Trading Using Regulatory News Volume as a Leading Indicator*

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![API Source](https://img.shields.io/badge/Bright%20Data-SERP%20API-orange.svg)](https://brightdata.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Track 2 — Finance & Market Intelligence**  
*Built for the Web Data UNLOCKED Hackathon — lablab.ai × Bright Data*

---

## 💡 The Core Hypothesis

In pairs trading, statistical cointegration breakdown is typically discovered **ex-post**: a spread diverges, the Ornstein-Uhlenbeck half-life doubles, the portfolio takes a drawdown, and the trading system is forced to cut the loss.

**RegimeWatch is designed to detect statistical breakdown ex-ante.**

We hypothesize that asymmetric regulatory pressure (e.g., SEC probes, class-action lawsuits, policy changes, anti-trust investigations) on one leg of a cointegrated pair degrades its structural covariance before the price charts reflect the change. 

By scraping daily search queries through **Bright Data's high-throughput SERP API**, we build a **Regulatory Attention Index (RAI)** and apply **Granger Causality testing** to determine if divergence in regulatory attention mathematically leads (causes) an increase in Ornstein-Uhlenbeck half-life (slower mean reversion).

---

## ⚙️ Features & Capabilities

- **🚀 Concurrency Engine**: Fetches and parses global regulatory news scores in parallel using a multi-threaded pool (`ThreadPoolExecutor`), compressing a 2-hour sequential search query pipeline into **under 4 minutes**.
- **📈 Advanced Quant Engine**:
  - **Johansen Cointegration Rank**: Rolling trace statistic calculations to monitor the structural rank of a two-asset system.
  - **Ornstein-Uhlenbeck (OU) Half-Life**: OLS estimation of spread mean-reversion speed via differential lag-1 regression.
  - **Granger Causality Tests**: Runs formal linear causality tests across lags 1 to 5 to prove the leading property of the RAI divergence index.
- **🧪 Custom Ticker Pair Analyzer (Sandbox)**: An interactive portal letting judges input **any arbitrary tickers** (e.g., `AAPL` vs `MSFT`, `KO` vs `PEP`) to fetch live prices, scrape concurrent regulatory news, test cointegration, and generate live health objects on the fly in **~15 seconds**.
- **📊 Premium Streamlit Dashboard**: Clean dark-mode visualization of dual-axis charts, causality p-value tables, and automated risk scoring (`NORMAL`, `WATCH`, `ELEVATED`, `SUSPEND`).

---

## 🏗️ Architecture & Data Pipeline

```
          [ Yahoo Finance API ]              [ Bright Data SERP API ]
                    │                                    │
           (Historical Prices)                   (News SERP Scrapes)
                    │                                    │
                    ▼                                    ▼
       ┌────────────────────────┐           ┌────────────────────────┐
       │   Pairs Engine (OU)    │           │    RAI Signal Engine   │
       │  • Johansen Rank & eig │           │  • Parallel Requests   │
       │  • Rolling Half-Life   │           │  • Z-Score Normalised  │
       └───────────┬────────────┘           └────────────┬───────────┘
                   │                                     │
                   │           (Spread Metrics)          │
                   └─────────────────┬───────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   Hypothesis Engine    │
                        │  • Granger Causality   │
                        │  • Lag Significance    │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │  Output: Health Objects│
                        │  • JSON Serialization  │
                        │  • Risk Flags Assigned │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │  Streamlit Dashboard   │
                        │  • Multi-Axis Graphs   │
                        │  • Live Ticker Sandbox │
                        └────────────────────────┘
```

---

## 📂 Project Structure

```
regimewatch/
├── main.py                  # CLI Orchestrator: runs full preset pipeline
├── config.py                # Parameters, default pairs, and ticker name maps
├── data/
│   ├── fetch_prices.py      # yfinance downloader (resilient to MultiIndexes)
│   └── fetch_rai.py         # Concurrent Bright Data SERP API news fetcher & sentiment engine
├── engine/
│   ├── pairs.py             # Rolling Johansen rank, beta hedges, and OU half-life
│   ├── rai.py               # Spreads raw RAI to absolute pair divergence signals
│   └── granger.py           # Linear regression causality tests
├── output/
│   ├── health.py            # Pair Health Object builder & dynamic risk flags
│   └── schema.py            # Dataclass definitions for structured output JSONs
├── dashboard/
│   └── app.py               # Streamlit application & interactive Sandbox UI
├── results/                 # Runtime caching directory (pre-computed CSVs/JSONs)
├── requirements.txt         # Package dependencies
└── README.md                # System documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone & Install
Ensure you have Python 3.10+ installed.
```bash
git clone https://github.com/YOUR_USERNAME/RegimeWatch.git
cd RegimeWatch
pip install -r regimewatch/requirements.txt
```

### 2. Configure Credentials
Create a `.env` file inside the `regimewatch/` directory:
```env
BRIGHTDATA_API_KEY=your_brightdata_api_key_here
BRIGHTDATA_SERP_ZONE=serp_api1
```

---

## 🚀 Running the Project

### A. Run the Live Streamlit Dashboard (Recommended)
This launches the interactive visualization. Judges can use the preset data or use the custom analyzer inside the sidebar:
```bash
streamlit run regimewatch/dashboard/app.py
```

### B. Standalone CLI Execution
To rebuild the entire pre-computed pipeline datasets for all 10 default preset pairs:
```bash
python regimewatch/main.py
```

---

## 🧪 Scientific & Mathematical Framing

### 1. Cointegration & Half-Life
For an asset pair $(A, B)$ with log-prices $p_t^A$ and $p_t^B$, we compute the OLS hedge ratio $\beta$:
$$p_t^A = \mu + \beta p_t^B + \epsilon_t$$

Where $\epsilon_t$ represents the mean-reverting portfolio spread. To quantify the speed of mean reversion, we model the spread $\epsilon_t$ as a continuous Ornstein-Uhlenbeck process:
$$d\epsilon_t = \theta(\mu - \epsilon_t)dt + \sigma dW_t$$

Regressing the discrete differences $\Delta \epsilon_t = \lambda \epsilon_{t-1} + u_t$, the half-life of mean reversion ($HL$) is:
$$HL = -\frac{\ln(2)}{\lambda}$$

*A high half-life indicates slow mean reversion (weakening cointegration pull), while an undefined half-life ($\lambda \ge 0$) signals structural breakdown.*

### 2. Regulatory Attention Index (RAI)
For each company $i$, the raw daily signal is built on search volumes containing targeted regulatory, legal, and compliance keywords. A naive negative sentiment penalty is applied:
$$RAI_{i, t} = \text{Count} + (\text{Negative Sentiment Score} \times 0.5)$$

The raw signal is z-score normalized to yield $RAI_{norm}$. The absolute divergence for a pair is defined as:
$$RAI\_Divergence_t = |RAI_{norm, t}^A - RAI_{norm, t}^B|$$

### 3. Granger Causality Hypothesis
To verify if RAI divergence acts as a leading indicator, we fit a Vector Autoregression (VAR) and test the null hypothesis:
$$H_0: \text{RAI Divergence does not Granger-cause increases in OU Half-Life.}$$

We verify this by analyzing F-statistics and p-values across lags $1 \le k \le 5$.

---

## 🛡️ Risk States Mapping
Pair health is evaluated daily and categorized into four actionable risk states:
- **`NORMAL` (Green)**: Cointegration is healthy; spread is highly mean-reverting.
- **`WATCH` (Yellow)**: RAI divergence z-score exceeds `0.8` $\sigma$.
- **`ELEVATED` (Orange)**: RAI divergence exceeds `1.5` $\sigma$ OR spread shows significant statistical slowing ($\text{Half-Life Slope} > 2.0$).
- **`SUSPEND` (Red)**: Extreme divergence exceeds `2.2` $\sigma$. **Algorithms should pause all position entry.**

---

## 💎 Bright Data Integration

RegimeWatch utilizes the **Bright Data SERP API** in the following ways:
1. **High-Throughput Concurrent Scrapes**: Distributes parallel requests through the Bright Data SERP zone to fetch search results for 20 companies across 20 trading dates simultaneously.
2. **Proxy Zones**: Leverages Bright Data's proxy network to avoid rate limits, bypass CAPTCHAs, and deliver robust daily news counts.
3. **Naively Structured Payload**: Targets Google News index parameters using the `parsed_light` JSON contract.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*RegimeWatch is built for the Web Data UNLOCKED Hackathon by lablab.ai × Bright Data.*
