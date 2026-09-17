# -*- coding: utf-8 -*-
# ==============================================================================
# IT3091 Machine Learning Assignment - Component 1
# Component Lead: Member 1 (Data Acquisition, Cleaning & Exploratory Data Analysis)
# Track: Industry Explorer | Lens: Investment Decision Support
# Target: HDFC Bank Next-Day Price Movement Direction (Binary Classification)
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils import set_seed, setup_plotting_style, RAW_DATA_PATH, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR

def load_and_clean_raw_data(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Parses raw HDFC Bank historical dataset, fixes multi-index/ticker header artifacts,
    sorts chronologically, resolves missing trading volumes, and formats datatypes.
    """
    set_seed(42)
    print("======================================================================")
    print("COMPONENT 1: DATA CLEANING & EXPLORATORY DATA ANALYSIS (EDA)")
    print("======================================================================\n")
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    # Read raw CSV skipping ticker metadata rows
    df = pd.read_csv(raw_path, skiprows=[1, 2])
    
    # Standardize column naming
    rename_dict = {}
    for col in df.columns:
        c_clean = str(col).strip()
        if 'price' in c_clean.lower() or 'date' in c_clean.lower():
            rename_dict[col] = 'Date'
        elif 'adj' in c_clean.lower():
            rename_dict[col] = 'Adj_Close'
        elif 'close' in c_clean.lower():
            rename_dict[col] = 'Close'
        elif 'open' in c_clean.lower():
            rename_dict[col] = 'Open'
        elif 'high' in c_clean.lower():
            rename_dict[col] = 'High'
        elif 'low' in c_clean.lower():
            rename_dict[col] = 'Low'
        elif 'volume' in c_clean.lower():
            rename_dict[col] = 'Volume'
            
    df = df.rename(columns=rename_dict)
    
    # Ensure required OHLCV columns exist
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for req in required_cols:
        if req not in df.columns:
            raise KeyError(f"Required column '{req}' missing from dataset.")
            
    # Parse dates and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # Convert numerical columns
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Adj_Close' in df.columns:
        num_cols.append('Adj_Close')
        
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Forward-fill and backward-fill any isolated missing market prices
    df[num_cols] = df[num_cols].ffill().bfill()
    
    # Handle zero-volume anomalies (e.g. trading halts or recording artifacts)
    df['Volume'] = df['Volume'].astype(float)
    zero_vol_mask = (df['Volume'] <= 0) | df['Volume'].isna()
    if zero_vol_mask.sum() > 0:
        rolling_median_vol = df['Volume'].rolling(20, min_periods=1).median()
        df.loc[zero_vol_mask, 'Volume'] = rolling_median_vol.loc[zero_vol_mask]
        df['Volume'] = df['Volume'].fillna(df['Volume'].median())
        
    # Calculate daily simple and log returns for statistical profiling
    df['daily_return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    out_path = os.path.join(PROCESSED_DATA_DIR, 'hdfc_cleaned.csv')
    df.to_csv(out_path, index=False)
    
    print(f"Successfully cleaned {len(df):,d} trading sessions from {df['Date'].dt.date.iloc[0]} to {df['Date'].dt.date.iloc[-1]}.")
    print(f"Cleaned dataset saved: {out_path}\n")
    return df

def generate_eda_visualizations_and_summary(df: pd.DataFrame):
    """Generates professional EDA figures and summary tables."""
    setup_plotting_style()
    print("Generating EDA visualization figures...")
    
    # Figure 1: 30-Year Price & Volume History
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(df['Date'], df['Close'], color='#1f77b4', lw=1.5, label='HDFC Bank Close Price (INR)')
    ax1.set_title('HDFC Bank Historical Stock Price & Trading Volume (1996 - 2026)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Stock Price (INR)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(df['Date'], df['Volume'] / 1e6, color='#ff7f0e', alpha=0.6, width=2.0, label='Volume (Millions)')
    ax2.set_ylabel('Volume (M)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '01_eda_price_volume_history.png'))
    plt.close()
    
    # Figure 2: Return Distribution & Normality (Fat Tails)
    clean_rets = df['daily_return'].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(clean_rets, bins=100, kde=True, color='#2ca02c', ax=ax1, stat='density', label='Empirical Distribution')
    mu, sigma = clean_rets.mean(), clean_rets.std()
    x = np.linspace(clean_rets.min(), clean_rets.max(), 500)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', lw=2, label=f'Gaussian Normal Fit (mu={mu:.4f}, sigma={sigma:.4f})')
    ax1.set_title('Daily Return Distribution vs. Normal Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Density')
    ax1.set_xlim([-0.10, 0.10])
    ax1.legend()
    
    stats.probplot(clean_rets, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Empirical Daily Returns vs. Normal Quantiles', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '02_eda_return_distribution.png'))
    plt.close()
    
    # Figure 3: Volatility Regimes
    df['vol_30d'] = df['daily_return'].rolling(30).std() * np.sqrt(252) * 100
    df['vol_90d'] = df['daily_return'].rolling(90).std() * np.sqrt(252) * 100
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Date'], df['vol_30d'], label='30-Day Rolling Annualized Volatility (%)', color='#9467bd', lw=1.2, alpha=0.8)
    ax.plot(df['Date'], df['vol_90d'], label='90-Day Rolling Annualized Volatility (%)', color='#d62728', lw=1.8)
    ax.axhline(df['vol_30d'].mean(), color='black', linestyle=':', label=f'Historical Mean ({df["vol_30d"].mean():.1f}%)')
    ax.set_title('HDFC Bank Rolling Annualized Volatility Regimes (1996 - 2026)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '03_eda_volatility_regimes.png'))
    plt.close()
    
    # Figure 4: Raw Price and Volume Correlation Heatmap
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return']
    corr_matrix = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title('Correlation Heatmap: Raw Market Features & Daily Returns', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '04_eda_correlation_heatmap.png'))
    plt.close()
    
    # Save Statistical Summary
    summary_stats = {
        'Metric': [
            'Total Trading Sessions', 'Start Date', 'End Date',
            'Mean Daily Return (%)', 'Annualized Return (%)', 'Daily Volatility (%)',
            'Annualized Volatility (%)', 'Skewness', 'Excess Kurtosis',
            'Max Single-Day Gain (%)', 'Max Single-Day Loss (%)',
            'Positive Return Days (UP %)', 'Negative Return Days (DOWN %)'
        ],
        'Value': [
            f"{len(df):,d}", str(df['Date'].dt.date.iloc[0]), str(df['Date'].dt.date.iloc[-1]),
            f"{clean_rets.mean()*100:.4f}%", f"{((1+clean_rets.mean())**252 - 1)*100:.2f}%",
            f"{clean_rets.std()*100:.4f}%", f"{clean_rets.std()*np.sqrt(252)*100:.2f}%",
            f"{clean_rets.skew():.4f}", f"{clean_rets.kurtosis():.4f}",
            f"{clean_rets.max()*100:.2f}% (on {df.loc[df['daily_return'].idxmax(), 'Date'].strftime('%Y-%m-%d')})",
            f"{clean_rets.min()*100:.2f}% (on {df.loc[df['daily_return'].idxmin(), 'Date'].strftime('%Y-%m-%d')})",
            f"{(clean_rets > 0).mean()*100:.2f}% ({(clean_rets > 0).sum():,d} days)",
            f"{(clean_rets <= 0).mean()*100:.2f}% ({(clean_rets <= 0).sum():,d} days)"
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(REPORTS_DIR, 'eda_summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"EDA figures generated in reports/figures/")
    print(f"EDA summary stats saved: {summary_path}\n")
    return summary_df

if __name__ == '__main__':
    cleaned_df = load_and_clean_raw_data()
    generate_eda_visualizations_and_summary(cleaned_df)
