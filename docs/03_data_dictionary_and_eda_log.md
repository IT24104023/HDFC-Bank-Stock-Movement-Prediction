# Document 3: Data Dictionary & Exploratory Data Analysis Log

## 1. Raw Dataset Overview
The raw dataset comprises 30+ years of daily equity market transactions for **HDFC Bank Limited** (`HDFCBANK.NS`), traded on the National Stock Exchange of India (NSE).
- **Source:** Yahoo Finance historical equity data (`hdfc_bank_historical_data.csv`).
- **Temporal Span:** January 1, 1996 to September 11, 2026.
- **Raw Observations:** 7,709 trading sessions.

---

## 2. Data Dictionary

### Raw Market Features
| Column Name | Data Type | Physical Meaning | Unit / Scale | Validation Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `Date` | `datetime64[ns]` | Trading session date | `YYYY-MM-DD` | Unique, strictly ascending, non-null |
| `Open` | `float64` | Opening price at market open (09:15 IST) | Indian Rupee (INR) | $> 0.0$, $	ext{Low} \le 	ext{Open} \le 	ext{High}$ |
| `High` | `float64` | Highest intraday traded price | Indian Rupee (INR) | $\ge \max(	ext{Open}, 	ext{Close}, 	ext{Low})$ |
| `Low` | `float64` | Lowest intraday traded price | Indian Rupee (INR) | $\le \min(	ext{Open}, 	ext{Close}, 	ext{High})$ |
| `Close` | `float64` | Final settlement price at market close (15:30 IST) | Indian Rupee (INR) | $> 0.0$, $	ext{Low} \le 	ext{Close} \le 	ext{High}$ |
| `Adj_Close`| `float64` | Corporate actions adjusted close price | Indian Rupee (INR) | $> 0.0$ |
| `Volume` | `float64` | Total number of equity shares traded | Share Count | $> 0$ (imputed for 121 zero sessions) |

---

## 3. Data Cleaning & Transformation Pipeline
1. **Multi-Row Header Standardization:** Parsed Yahoo Finance header quirks where row 0 contained price tags, row 1 contained ticker symbols (`HDFCBANK.NS`), and row 2 contained `Date`. Mapped first column to `Date` and standard OHLCV columns.
2. **Chronological Sorting:** Verified strict monotonically increasing order of dates from 1996-01-01 to 2026-09-11.
3. **Missing Value Resolution:** Zero missing entries detected in price columns after standard forward/back-fill.
4. **Volume Imputation:** 121 historical zero-volume trading sessions (0.015% of total) replaced with 20-day rolling medians.
5. **Output Cleaned File:** Persisted to `data/processed/hdfc_cleaned.csv` (7,709 rows).

---

## 4. Exploratory Data Analysis & Statistical Profiling

### Summary Statistics (30-Year History)
| Metric | Value | Financial Interpretation |
| :--- | :--- | :--- |
| **Total Trading Sessions** | 7,709 | 30.7 years of continuous market observations |
| **Date Range** | 1996-01-01 to 2026-09-11 | Spans Asian Financial Crisis, 2008 GFC, 2020 COVID shock, HDFC-HDFC Bank merger |
| **Mean Daily Simple Return** | $+0.1206\%$ | Positive long-term equity growth trajectory |
| **Annualized Compounded Return** | $+22.58\%$ | Multi-decade compound annual growth rate (CAGR) |
| **Daily Volatility ($\sigma$)** | $1.8687\%$ | Standard deviation of daily percentage price changes |
| **Annualized Volatility** | $29.66\%$ | Annualized historical risk ($\sigma 	imes \sqrt{252}$) |
| **Skewness** | $+0.5975$ | Moderate positive skew (intermittent large upside rallies) |
| **Excess Kurtosis** | $+8.2144$ | **Leptokurtic (Heavy Fat Tails):** Returns deviate significantly from Gaussian Normal |
| **Max Single-Day Gain** | $+20.00\%$ | Upper circuit hit on 2009-05-18 (General Election post-results rally) |
| **Max Single-Day Loss** | $-18.06\%$ | Severe market crash on 2004-05-17 (Black Monday political shock) |
| **UP Days ($y_t = 1$)** | $3,833 	ext{ days } (49.72\%)$ | Highly balanced binary target distribution |
| **DOWN Days ($y_t = 0$)** | $3,876 	ext{ days } (50.28\%)$ | Confirms absence of majority-class target imbalance |

---

## 5. Visual EDA Artifacts Summary

1. **`01_eda_price_volume_history.png`:**
   - Highlights 30-year exponential growth of HDFC Bank, transitioning from single-digit base to peak valuation $> 1,700$ INR.
   - Volume displays distinct structural surges during major market events (2008 GFC, 2020 pandemic, 2023 parent merger).
2. **`02_eda_return_distribution.png`:**
   - Empirical density histogram versus theoretical Gaussian normal curve.
   - Q-Q Plot definitively demonstrates severe fat-tail departures in extreme quantiles (leptokurtosis), demonstrating why linear models require regularization and non-linear tree ensembles are necessary.
3. **`03_eda_volatility_regimes.png`:**
   - 30-day and 90-day rolling annualized volatility.
   - Highlights clustering of extreme volatility during 1998 Asian Crisis ($>60\%$), 2008 GFC ($>55\%$), and March 2020 COVID shock ($>65\%$), subsiding into lower-volatility regimes ($18-25\%$) during bull markets.
4. **`04_eda_correlation_heatmap.png`:**
   - Raw OHLC prices exhibit extreme collinearity ($r > 0.999$), proving that raw price levels cannot be used directly as ML features.
   - Motivates stationary price transformations (returns, normalized oscillator bounds, moving average ratios).
