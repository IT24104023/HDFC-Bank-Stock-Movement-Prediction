# -*- coding: utf-8 -*-
# ==============================================================================
# IT3091 Machine Learning Assignment - Component 2
# Component Lead: Member 2 (Feature Engineering & Preprocessing)
# Track: Industry Explorer | Lens: Investment Decision Support
# Target: HDFC Bank Next-Day Price Movement Direction (Binary Classification)
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils import set_seed, FEATURE_COLUMNS, TARGET_COLUMN

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI) using Wilder smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) normalized volatility measure."""
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers technical indicators, momentum, volatility, and volume features
    strictly at time t to predict direction at t+1 without lookahead leakage.
    """
    df = df.copy()
    c = df['Close']
    o = df['Open']
    h = df['High']
    l = df['Low']
    v = df['Volume']
    
    # 1. Price Momentum & Lagged Returns
    df['return_1d'] = c.pct_change(1)
    df['return_2d'] = c.pct_change(2)
    df['return_3d'] = c.pct_change(3)
    df['return_5d'] = c.pct_change(5)
    df['return_10d'] = c.pct_change(10)
    df['return_20d'] = c.pct_change(20)
    
    # 2. Moving Average Trend Ratios
    for w in [5, 10, 20, 50, 200]:
        df[f'sma_{w}_ratio'] = c / c.rolling(w).mean() - 1.0
        
    for w in [12, 26]:
        df[f'ema_{w}_ratio'] = c / c.ewm(span=w, adjust=False).mean() - 1.0
        
    df['sma_5_20_ratio'] = c.rolling(5).mean() / c.rolling(20).mean() - 1.0
    df['sma_50_200_ratio'] = c.rolling(50).mean() / c.rolling(200).mean() - 1.0
    
    # 3. Oscillators & Momentum
    df['rsi_14'] = calculate_rsi(c, 14)
    
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    macd_line = (ema_12 - ema_26) / c
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_diff'] = macd_line - macd_signal
    
    # Stochastic Oscillator
    low_14 = l.rolling(14).min()
    high_14 = h.rolling(14).max()
    df['stoch_k_14'] = ((c - low_14) / (high_14 - low_14 + 1e-10)) * 100.0
    df['stoch_d_3'] = df['stoch_k_14'].rolling(3).mean()
    
    # 4. Volatility & Price Action
    ret_1d = df['return_1d']
    df['volatility_20d'] = ret_1d.rolling(20).std() * np.sqrt(252)
    
    atr_14 = calculate_atr(df, 14)
    df['atr_14_ratio'] = atr_14 / c
    
    sma_20 = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    df['bb_percent_b'] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df['bb_bandwidth'] = (bb_upper - bb_lower) / sma_20
    
    df['hl_spread'] = (h - l) / c
    df['co_spread'] = (c - o) / (o + 1e-10)
    df['overnight_gap'] = (o - c.shift(1)) / (c.shift(1) + 1e-10)
    
    # 5. Volume Indicators
    df['vol_return_1d'] = v.pct_change(1)
    df['vol_sma_5_ratio'] = v / v.rolling(5).mean() - 1.0
    df['vol_sma_20_ratio'] = v / v.rolling(20).mean() - 1.0
    
    # 6. Calendar Features
    dates = pd.to_datetime(df['Date'])
    df['day_of_week'] = dates.dt.dayofweek
    df['month'] = dates.dt.month
    df['quarter'] = dates.dt.quarter
    
    # 7. Supervised Target Variable (Direction at t+1)
    # Target = 1 if Close(t+1) > Close(t), else 0
    next_close = c.shift(-1)
    df['next_return'] = (next_close - c) / c
    df['target'] = (next_close > c).astype(int)
    
    return df

def prepare_and_split_data(
    input_path: str = os.path.join(ROOT_DIR, 'data', 'processed', 'hdfc_cleaned.csv'),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """
    Loads cleaned data, computes features, drops warmup period and final unobserved target row,
    performs chronological splitting, and scales features strictly without data leakage.
    """
    set_seed(42)
    print("======================================================================")
    print("COMPONENT 2: FEATURE ENGINEERING & DATA SPLITTING")
    print("======================================================================\n")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data not found at {input_path}. Run Component 1 first.")
        
    df_raw = pd.read_csv(input_path)
    df_feat = compute_technical_features(df_raw)
    
    # Drop warm-up rows (200 rows needed for 200 SMA) and last row (target unknown)
    warmup_rows = 200
    df_valid = df_feat.iloc[warmup_rows:-1].copy().reset_index(drop=True)
    
    # Check nulls in engineered features
    feature_cols = [col for col in FEATURE_COLUMNS if col in df_valid.columns]
    df_valid = df_valid.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    
    n = len(df_valid)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    train_df = df_valid.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_valid.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = df_valid.iloc[n_train + n_val:].copy().reset_index(drop=True)
    
    print(f"Total valid samples: {n} (from {df_valid['Date'].iloc[0]} to {df_valid['Date'].iloc[-1]})")
    print(f"Train set:      {len(train_df):5d} samples ({train_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]})")
    print(f"Validation set: {len(val_df):5d} samples ({val_df['Date'].iloc[0]} to {val_df['Date'].iloc[-1]})")
    print(f"Test set:       {len(test_df):5d} samples ({test_df['Date'].iloc[0]} to {test_df['Date'].iloc[-1]})")
    
    # Verify class balance across splits
    print("\nClass balance (Target = 1 [UP] %):")
    print(f"  Train:      {train_df['target'].mean()*100:.2f}%")
    print(f"  Validation: {val_df['target'].mean()*100:.2f}%")
    print(f"  Test:       {test_df['target'].mean()*100:.2f}%")
    
    # Scale features using StandardScaler fitted ONLY on training data
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    # Save scaler
    scaler_path = os.path.join(ROOT_DIR, 'models', 'scaler.joblib')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler fitted strictly on training data and saved to: {scaler_path}")
    
    # Save feature datasets
    out_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    df_valid.to_csv(os.path.join(out_dir, 'features_full.csv'), index=False)
    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    print(f"Feature datasets saved in: {out_dir}")
    print(f"Engineered {len(feature_cols)} technical features across price, momentum, volatility, and volume.\n")
    return train_df, val_df, test_df, scaler, feature_cols

if __name__ == '__main__':
    prepare_and_split_data()
