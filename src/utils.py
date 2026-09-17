# -*- coding: utf-8 -*-
# ==============================================================================
# IT3091 Machine Learning Assignment - Utilities & Common Helpers
# Track: Industry Explorer | Lens: Investment Decision Support
# Target: HDFC Bank Next-Day Price Movement Direction (Binary Classification)
# ==============================================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, roc_curve, log_loss
)

RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'hdfc_bank_historical_data.csv')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

FEATURE_COLUMNS = [
    'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
    'sma_5_ratio', 'sma_10_ratio', 'sma_20_ratio', 'sma_50_ratio', 'sma_200_ratio',
    'ema_12_ratio', 'ema_26_ratio', 'sma_5_20_ratio', 'sma_50_200_ratio',
    'rsi_14', 'macd_line', 'macd_signal', 'macd_diff', 'stoch_k_14', 'stoch_d_3',
    'volatility_20d', 'atr_14_ratio', 'bb_percent_b', 'bb_bandwidth',
    'hl_spread', 'co_spread', 'overnight_gap',
    'vol_return_1d', 'vol_sma_5_ratio', 'vol_sma_20_ratio',
    'day_of_week', 'month', 'quarter'
]
TARGET_COLUMN = 'target'

def set_seed(seed=RANDOM_SEED):
    """Sets random seeds across random, numpy, and os environment for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

for d in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

def setup_plotting_style():
    """Configures consistent seaborn and matplotlib publication styling."""
    sns.set_theme(style='whitegrid', palette='muted')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

class NaiveMomentumBaseline(BaseEstimator, ClassifierMixin):
    """
    Rule-based financial baseline:
    Predicts next-day direction as UP (1) if today's 1-day return was positive,
    otherwise DOWN (0).
    """
    def __init__(self, return_feature_idx=0):
        self.return_feature_idx = return_feature_idx
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        ret_1d = X_arr[:, self.return_feature_idx]
        return (ret_1d > 0).astype(int)

    def predict_proba(self, X):
        preds = self.predict(X)
        probs = np.zeros((len(preds), 2))
        probs[:, 1] = np.where(preds == 1, 0.55, 0.45)
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs

def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates full statistical and classification metrics:
    Accuracy, UP/DOWN Precision, UP/DOWN Recall, UP/DOWN F1, Macro F1,
    ROC-AUC, Brier Score, and Log Loss.
    """
    metrics = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision_UP': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'Recall_UP': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'F1_UP': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'Precision_DOWN': float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'Recall_DOWN': float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'F1_DOWN': float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'Macro_F1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics['ROC_AUC'] = float(roc_auc_score(y_true, y_prob))
            metrics['Brier_Score'] = float(brier_score_loss(y_true, y_prob))
            metrics['Log_Loss'] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics['ROC_AUC'] = np.nan
            metrics['Brier_Score'] = np.nan
            metrics['Log_Loss'] = np.nan
    else:
        metrics['ROC_AUC'] = np.nan
        metrics['Brier_Score'] = np.nan
        metrics['Log_Loss'] = np.nan
    return metrics
