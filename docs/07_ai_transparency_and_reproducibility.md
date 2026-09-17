# Document 7: AI Transparency, Tool Usage & Reproducibility Statement

## 1. AI Tool Usage Declaration
In accordance with academic integrity guidelines and the IT3091 Machine Learning assignment policy, this declaration documents the use of AI assistance tools during the development of this project.

- **Primary AI Assistant:** Google Antigravity / Gemini Coding Assistant.
- **Scope of AI Assistance:**
  1. *Code Scaffolding & Architecture:* Structuring modular Python scripts, plotting functions, and pipeline orchestration.
  2. *Documentation Drafting:* Assisting in formatting LaTeX equations, markdown tables, and Mermaid architecture diagrams.
  3. *Syntax Debugging:* Resolving Windows PowerShell unicode and encoding quirks.
- **Human Group Review & Verification:** All mathematical formulas, data cleaning logic, feature calculations, time-series splitting boundaries, model training pipelines, and financial interpretations were rigorously designed, verified, and audited by the 4 group members.

---

## 2. Reproducibility Guarantee & Seed Configuration
Exact reproducibility across all environments is ensured through:
- **Global Seed Configuration:** `RANDOM_SEED = 42` set across Python `random`, `numpy.random`, `os.environ['PYTHONHASHSEED']`, and scikit-learn estimators.
- **Zero Lookahead Leakage:** Scaler parameters ($\mu, \sigma$) fitted strictly on training partition and persisted to `models/scaler.joblib`.
- **Environment & Dependency Manifest:** Standardized Python 3.13 dependencies specified in `requirements.txt`.

---

## 3. End-to-End Replication Protocol

To replicate all results from scratch in a clean environment:

```bash
# 1. Clone repository and navigate to root directory
cd "c:/Users/moham/Downloads/Machine Learning"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute master end-to-end pipeline (All 4 components)
python run_pipeline.py

# 4. (Optional) Run Jupyter Notebook
jupyter notebook notebooks/HDFC_Bank_ML_Assignment_Master.ipynb
```
