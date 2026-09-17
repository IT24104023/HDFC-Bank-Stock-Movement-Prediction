# -*- coding: utf-8 -*-
# ==============================================================================
# IT3091 Machine Learning Assignment - Master Pipeline Orchestrator
# Project: HDFC Bank Stock Price Movement Prediction Using Historical Market Data
# Track: Industry Explorer | Lens: Investment Decision Support
# Team: 4-Member Project Group
# ==============================================================================

import os
import sys
import time
import importlib

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import components
c1 = importlib.import_module('src.01_data_cleaning_and_eda')
c2 = importlib.import_module('src.02_feature_engineering')
c3 = importlib.import_module('src.03_model_training')
c4 = importlib.import_module('src.04_evaluation')

def run_end_to_end_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("  HDFC BANK STOCK PRICE MOVEMENT PREDICTION PIPELINE")
    print("  Course: IT3091 Machine Learning | Track: Industry Explorer (Decision Support)")
    print("=" * 80)
    print("\n[1/4] EXECUTING COMPONENT 1: Data Cleaning & Exploratory Data Analysis...")
    t1 = time.time()
    cleaned_df = c1.load_and_clean_raw_data()
    c1.generate_eda_visualizations_and_summary(cleaned_df)
    print(f"[OK] Component 1 completed in {time.time() - t1:.2f}s")
    
    print("\n[2/4] EXECUTING COMPONENT 2: Feature Engineering & Temporal Splitting...")
    t2 = time.time()
    train_df, val_df, test_df, scaler, feature_cols = c2.prepare_and_split_data()
    print(f"[OK] Component 2 completed in {time.time() - t2:.2f}s")
    
    print("\n[3/4] EXECUTING COMPONENT 3: Model Training, Tuning & Baseline Benchmarking...")
    t3 = time.time()
    models, val_metrics = c3.train_and_validate_models()
    print(f"[OK] Component 3 completed in {time.time() - t3:.2f}s")
    
    print("\n[4/4] EXECUTING COMPONENT 4: Model Evaluation, Explainability & Decision Support...")
    t4 = time.time()
    test_metrics, sim_metrics = c4.evaluate_models_on_test_set()
    print(f"[OK] Component 4 completed in {time.time() - t4:.2f}s")
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"  PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 80)
    print("\nSummary of Generated Artifacts:")
    print("  - Cleaned & Feature Datasets: data/processed/")
    print("  - Serialized Trained Models:  models/")
    print("  - Evaluation Figures:         reports/figures/")
    print("  - Evaluation CSV Tables:      reports/")
    print("=" * 80)

if __name__ == '__main__':
    run_end_to_end_pipeline()
