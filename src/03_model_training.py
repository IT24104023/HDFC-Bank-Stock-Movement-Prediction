# -*- coding: utf-8 -*-
# ==============================================================================
# IT3091 Machine Learning Assignment - Component 3
# Component Lead: Member 3 (Model Training, Tuning & Baseline Benchmarking)
# Track: Industry Explorer | Lens: Investment Decision Support
# Target: HDFC Bank Next-Day Price Movement Direction (Binary Classification)
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils import (
    set_seed, calculate_comprehensive_metrics, NaiveMomentumBaseline,
    FEATURE_COLUMNS, TARGET_COLUMN, MODELS_DIR, REPORTS_DIR
)

def train_and_validate_models():
    """
    Trains baseline and 4 supervised classification models on training data,
    evaluates on the validation set, and persists trained artifacts.
    """
    set_seed(42)
    print("======================================================================")
    print("COMPONENT 3: MODEL TRAINING & HYPERPARAMETER TUNING")
    print("======================================================================\n")
    
    # Load train and val data
    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    
    # Scale feature matrices
    X_train = scaler.transform(train_df[feature_cols])
    y_train = train_df[TARGET_COLUMN].values
    
    X_val = scaler.transform(val_df[feature_cols])
    y_val = val_df[TARGET_COLUMN].values
    
    ret1d_idx = feature_cols.index('return_1d') if 'return_1d' in feature_cols else 0
    
    # Initialize models with tuned hyperparameters
    models = {
        'Baseline_Momentum': NaiveMomentumBaseline(return_feature_idx=ret1d_idx),
        'Logistic_Regression': LogisticRegression(
            C=0.10, penalty='l2', solver='lbfgs', class_weight='balanced',
            max_iter=1000, random_state=42
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'Hist_Gradient_Boosting': HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=150, max_depth=4,
            min_samples_leaf=25, l2_regularization=0.1, random_state=42
        ),
        'Support_Vector_Classifier': SVC(
            C=0.5, kernel='rbf', gamma='scale', probability=True, random_state=42
        )
    }
    
    val_results = []
    
    print(f"Training {len(models)} models on {len(X_train)} training sessions...\n")
    
    for name, model in models.items():
        print(f"--> Training [{name}]...")
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = calculate_comprehensive_metrics(y_val, y_val_pred, y_val_prob)
        metrics['Model'] = name
        val_results.append(metrics)
        
        # Save model artifact
        save_path = os.path.join(MODELS_DIR, f'{name}.joblib')
        joblib.dump(model, save_path)
        print(f"    Validation Accuracy: {metrics['Accuracy']*100:.2f}% | Macro F1: {metrics['Macro_F1']:.4f} | ROC-AUC: {metrics['ROC_AUC']:.4f}")
        print(f"    Saved model to: {save_path}")
        
    val_df_metrics = pd.DataFrame(val_results)[['Model', 'Accuracy', 'Macro_F1', 'ROC_AUC', 'Precision_UP', 'Recall_UP', 'F1_UP', 'Brier_Score']]
    val_metrics_path = os.path.join(REPORTS_DIR, 'validation_metrics.csv')
    val_df_metrics.to_csv(val_metrics_path, index=False)
    
    print(f"\nValidation performance summary saved to: {val_metrics_path}\n")
    return models, val_df_metrics

if __name__ == '__main__':
    train_and_validate_models()
