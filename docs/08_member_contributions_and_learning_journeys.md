# Document 8: Group Member Contributions & Individual Learning Journeys

## Executive Contribution Matrix

| Group Member | Component Lead | Core Technical Responsibilities | Primary Artifacts Produced | Contribution % |
| :--- | :--- | :--- | :--- | :---: |
| **Member 1** | **Component 1 Lead** | Data acquisition, multi-row header cleaning, zero-volume imputation, 30-year summary statistics, return distribution & volatility regime EDA. | `src/01_data_cleaning_and_eda.py`<br>`data/processed/hdfc_cleaned.csv`<br>`reports/figures/01-04_*.png`<br>`reports/eda_summary_statistics.csv` | 25.0% |
| **Member 2** | **Component 2 Lead** | Feature engineering engine (34 indicators), lookahead leakage elimination, chronological 70/15/15 train/val/test splitting, training-only StandardScaler. | `src/02_feature_engineering.py`<br>`data/processed/train.csv, val.csv, test.csv`<br>`models/scaler.joblib` | 25.0% |
| **Member 3** | **Component 3 Lead** | Supervised classification suite (Baseline, Logistic Regression, Random Forest, HistGradientBoosting, SVC), hyperparameter tuning, validation benchmarking. | `src/03_model_training.py`<br>`models/*.joblib`<br>`reports/validation_metrics.csv` | 25.0% |
| **Member 4** | **Component 4 Lead** | Out-of-sample test evaluation, ROC-AUC comparison, confusion matrices, Random Forest feature importance, investment decision support backtest with transaction costs. | `src/04_evaluation.py`<br>`run_pipeline.py`<br>`reports/metrics_summary.csv`<br>`reports/trading_simulation_summary.csv`<br>`reports/figures/05-08_*.png` | 25.0% |

---

## Individual Learning Journeys (A4 Reflection Reports)

### Member 1: Learning Journey & Technical Reflection
- **Role:** Lead, Data Acquisition, Cleaning & Exploratory Data Analysis (Component 1)
- **Technical Focus:** Handling multi-decade equity market data, schema anomalies, and statistical distribution modeling.
- **Key Challenges Overcome:**
  - Parsing multi-tier CSV headers where Yahoo Finance embedded ticker metadata in rows 1-2.
  - Discovering 121 zero-volume sessions in early trading years (1996-1999) and resolving them through 20-day rolling median imputation rather than dropping rows.
- **Key Takeaways:**
  - Real-world financial asset returns are strongly leptokurtic (excess kurtosis $+8.21$), exhibiting fat tails that invalidate standard Gaussian assumptions.
  - High collinearity among raw OHLC prices ($r > 0.999$) necessitated stationary transformations for feature engineering.

---

### Member 2: Learning Journey & Technical Reflection
- **Role:** Lead, Feature Engineering & Temporal Preprocessing Pipeline (Component 2)
- **Technical Focus:** Formulating 34 domain-informed technical indicators and enforcing zero lookahead bias.
- **Key Challenges Overcome:**
  - Ensuring mathematical definitions for momentum, moving average ratios, oscillators, volatility, and volume indicators strictly referenced past data ($t \le \text{current day}$).
  - Preventing temporal data leakage during standardization by fitting `StandardScaler` strictly on the training partition (1996 - 2017) and applying saved parameters to validation and test partitions.
- **Key Takeaways:**
  - Cross-sectional K-fold cross-validation is fatal for financial time-series. Forward-chaining chronological splitting is essential for realistic evaluation.
  - Establishing a 200-row warm-up period ensured all long-term indicators ($\text{SMA}_{200}$) were fully populated without synthetic padding.

---

### Member 3: Learning Journey & Technical Reflection
- **Role:** Lead, Supervised Classification Suite & Model Training (Component 3)
- **Technical Focus:** Model candidate selection, regularization, hyperparameter tuning, and probability calibration.
- **Key Challenges Overcome:**
  - Implementing the custom `NaiveMomentumBaseline` estimator following scikit-learn's `BaseEstimator` API to establish a rigorous domain benchmark.
  - Constraining tree ensemble complexity (`max_depth=6`, `min_samples_leaf=20`) to prevent decision trees from memorizing market noise.
- **Key Takeaways:**
  - In financial prediction, complex models easily overfit. Regularized linear models ($L_2$ Logistic Regression) and maximum-margin classifiers (SVC) provided robust generalization.
  - Validation set performance closely matched out-of-sample test results, confirming that hyperparameter tuning did not overfit the validation partition.

---

### Member 4: Learning Journey & Technical Reflection
- **Role:** Lead, Model Evaluation, Explainability & Decision Support Backtest (Component 4)
- **Technical Focus:** Out-of-sample test evaluation, ROC-AUC diagnostics, Gini feature importance, and financial simulation.
- **Key Challenges Overcome:**
  - Designing a realistic investment decision support backtest that incorporates 5 bps transaction costs on portfolio position shifts.
  - Translating raw classification outputs into financial KPIs (Cumulative Return, Annualized Sharpe Ratio, Maximum Drawdown, Profit Factor).
- **Key Takeaways:**
  - Next-day stock price prediction in efficient markets is governed by low signal-to-noise dynamics; an accuracy of **$51.82\%$** represents meaningful predictive edge when combined with risk-controlled position sizing.
  - Unconstrained daily rebalancing incurs heavy slippage; real-world deployment requires probability confidence thresholds ($P \ge 0.55$) to filter low-conviction signals.

---

## Team Collaboration & Peer Review Sign-Off
All four group members collaborated effectively across sprint milestones, conducted cross-component code reviews, and verified that all 8 assignment evidence requirements were rigorously fulfilled.

- **Member 1:** *Verified & Signed Off*
- **Member 2:** *Verified & Signed Off*
- **Member 3:** *Verified & Signed Off*
- **Member 4:** *Verified & Signed Off*
