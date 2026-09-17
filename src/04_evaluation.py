# -*- coding: utf-8 -*-
# ==============================================================================
# IT3091 Machine Learning Assignment - Component 4
# Component Lead: Member 4 (Evaluation, Feature Explainability & Decision Support)
# Track: Industry Explorer | Lens: Investment Decision Support
# Target: HDFC Bank Next-Day Price Movement Direction (Binary Classification)
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils import (
    set_seed, setup_plotting_style, calculate_comprehensive_metrics, NaiveMomentumBaseline,
    FEATURE_COLUMNS, TARGET_COLUMN, MODELS_DIR, FIGURES_DIR, REPORTS_DIR
)

def evaluate_models_on_test_set():
    """
    Performs rigorous out-of-sample testing on the 15% holdout test set.
    Generates metrics table, ROC curve comparisons, and confusion matrices.
    """
    set_seed(42)
    setup_plotting_style()
    print("======================================================================")
    print("COMPONENT 4: MODEL EVALUATION, EXPLAINABILITY & DECISION SUPPORT")
    print("======================================================================\n")
    
    # Load test data
    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    feature_cols = [c for c in FEATURE_COLUMNS if c in test_df.columns]
    
    X_test = scaler.transform(test_df[feature_cols])
    y_test = test_df[TARGET_COLUMN].values
    
    model_names = [
        'Baseline_Momentum',
        'Logistic_Regression',
        'Random_Forest',
        'Hist_Gradient_Boosting',
        'Support_Vector_Classifier'
    ]
    
    models = {}
    test_metrics_list = []
    predictions_dict = {}
    probabilities_dict = {}
    
    print(f"Evaluating {len(model_names)} models on {len(test_df)} out-of-sample test sessions ({test_df['Date'].iloc[0]} to {test_df['Date'].iloc[-1]})...\n")
    
    for name in model_names:
        model_path = os.path.join(MODELS_DIR, f'{name}.joblib')
        model = joblib.load(model_path)
        models[name] = model
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        predictions_dict[name] = y_pred
        probabilities_dict[name] = y_prob
        
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_prob)
        metrics['Model'] = name
        test_metrics_list.append(metrics)
        
        print(f"--> {name:26s} | Acc: {metrics['Accuracy']*100:.2f}% | UP Prec: {metrics['Precision_UP']*100:.1f}% | UP Rec: {metrics['Recall_UP']*100:.1f}% | Macro F1: {metrics['Macro_F1']:.4f} | ROC-AUC: {metrics['ROC_AUC']:.4f}")
        
    metrics_df = pd.DataFrame(test_metrics_list)[
        ['Model', 'Accuracy', 'Precision_UP', 'Recall_UP', 'F1_UP', 'Precision_DOWN', 'Recall_DOWN', 'F1_DOWN', 'Macro_F1', 'ROC_AUC', 'Brier_Score', 'Log_Loss']
    ]
    metrics_summary_path = os.path.join(REPORTS_DIR, 'metrics_summary.csv')
    metrics_df.to_csv(metrics_summary_path, index=False)
    print(f"\nMetrics summary saved to: {metrics_summary_path}")
    
    # -------------------------------------------------------------
    # Figure 5: ROC Curve Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    for (name, model), color in zip(models.items(), colors):
        if name in probabilities_dict and probabilities_dict[name] is not None:
            y_prob = probabilities_dict[name]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name.replace("_", " ")} (AUC = {auc_val:.3f})')
            
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax.set_title('Out-of-Sample ROC Curves (Test Set: 2022 - 2026)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '05_model_roc_comparison.png'))
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 8: Confusion Matrix Multi-plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = confusion_matrix(y_test, predictions_dict[name])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred DOWN', 'Pred UP'],
                    yticklabels=['Actual DOWN', 'Actual UP'], ax=ax)
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)',
                        ha='center', va='center', color='darkgray', fontsize=10)
        ax.set_title(name.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_ylabel('True Direction')
        ax.set_xlabel('Predicted Direction')
        
    axes[5].axis('off')
    plt.suptitle('Out-of-Sample Confusion Matrices Across Models (Test Set)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '08_confusion_matrices.png'))
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 6: Feature Importance Analysis
    # -------------------------------------------------------------
    rf_model = models['Random_Forest']
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
    feat_imp_df.to_csv(os.path.join(REPORTS_DIR, 'feature_importance.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='Blues_r', ax=ax)
    ax.set_title('Top 15 Most Predictive Features (Random Forest MDI Gini Importance)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Decrease in Impurity (Gini Importance)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '06_feature_importance.png'))
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 7 & Table: Investment Decision Support Backtest Simulation
    # -------------------------------------------------------------
    print("Running Investment Decision Support Backtest Simulation...")
    next_returns = test_df['next_return'].values
    dates = pd.to_datetime(test_df['Date'])
    
    simulation_results = {}
    perf_metrics = []
    
    # 1. Buy & Hold Benchmark
    bh_daily_returns = next_returns
    bh_equity = (1.0 + pd.Series(bh_daily_returns)).cumprod()
    simulation_results['Buy_and_Hold'] = bh_equity
    
    rf_rate_daily = 0.065 / 252 # 6.5% Indian 10-Yr Sovereign Risk-Free Rate
    
    def calc_financial_kpis(strat_name, daily_rets, equity_curve):
        cum_ret = (equity_curve.iloc[-1] - 1.0) * 100
        ann_ret = ((equity_curve.iloc[-1]) ** (252 / len(daily_rets)) - 1.0) * 100
        ann_vol = daily_rets.std() * np.sqrt(252) * 100
        excess_rets = daily_rets - rf_rate_daily
        sharpe = (excess_rets.mean() / (daily_rets.std() + 1e-10)) * np.sqrt(252)
        
        # Max Drawdown
        roll_max = equity_curve.cummax()
        drawdown = (equity_curve - roll_max) / roll_max
        max_dd = drawdown.min() * 100
        
        trades = daily_rets != 0
        win_rate = (daily_rets[trades] > 0).mean() * 100 if trades.sum() > 0 else 0
        gains = daily_rets[daily_rets > 0].sum()
        losses = abs(daily_rets[daily_rets < 0].sum())
        profit_factor = (gains / (losses + 1e-10)) if losses > 0 else np.nan
        
        return {
            'Strategy': strat_name,
            'Cumulative_Return_%': cum_ret,
            'Annualized_Return_%': ann_ret,
            'Annualized_Vol_%': ann_vol,
            'Sharpe_Ratio (Rf=6.5%)': sharpe,
            'Max_Drawdown_%': max_dd,
            'Win_Rate_%': win_rate,
            'Profit_Factor': profit_factor
        }
        
    perf_metrics.append(calc_financial_kpis('Buy & Hold Benchmark', pd.Series(bh_daily_returns), bh_equity))
    
    tx_cost = 0.0005 # 5 bps transaction friction
    
    for name in model_names:
        preds = predictions_dict[name]
        pos = np.where(preds == 1, 1.0, 0.0) # Long or Cash
        pos_shifts = np.abs(np.diff(pos, prepend=pos[0]))
        strat_rets = pos * next_returns - (pos_shifts * tx_cost)
        equity = (1.0 + pd.Series(strat_rets)).cumprod()
        simulation_results[name] = equity
        perf_metrics.append(calc_financial_kpis(name.replace('_', ' '), pd.Series(strat_rets), equity))
        
    sim_df = pd.DataFrame(perf_metrics)
    sim_summary_path = os.path.join(REPORTS_DIR, 'trading_simulation_summary.csv')
    sim_df.to_csv(sim_summary_path, index=False)
    print(f"Trading simulation results saved to: {sim_summary_path}")
    
    # Plot Simulation Equity Curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(dates, simulation_results['Buy_and_Hold'], label='Buy & Hold Benchmark', color='black', linestyle='--', lw=1.8)
    for (name, color) in zip(model_names, colors):
        ax1.plot(dates, simulation_results[name], label=name.replace('_', ' '), color=color, lw=1.8)
        
    ax1.set_title('Investment Decision Support Backtest: Out-of-Sample Cumulative Wealth (2022 - 2026)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (Base = 1.0)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Drawdown for Buy & Hold vs Best Model
    best_model_name = 'Random_Forest'
    rf_eq = simulation_results[best_model_name]
    rf_dd = (rf_eq - rf_eq.cummax()) / rf_eq.cummax() * 100
    bh_dd = (bh_equity - bh_equity.cummax()) / bh_equity.cummax() * 100
    
    ax2.plot(dates, bh_dd, color='black', linestyle='--', label='Buy & Hold Drawdown (%)', lw=1.2)
    ax2.plot(dates, rf_dd, color='#2ca02c', label=f'{best_model_name} Drawdown (%)', lw=1.5)
    ax2.fill_between(dates, rf_dd, 0, color='#2ca02c', alpha=0.2)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '07_decision_support_simulation.png'))
    plt.close()
    
    print(f"Decision support simulation figure generated: reports/figures/07_decision_support_simulation.png\n")
    return metrics_df, sim_df

if __name__ == '__main__':
    evaluate_models_on_test_set()
