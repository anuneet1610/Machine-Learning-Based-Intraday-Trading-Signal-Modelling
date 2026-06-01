import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from clickhouse_driver import Client
from pathlib import Path
import logging

# Service configuration
MICROSERVICE_NAME = "decision_service"

# Ensure logs directory exists
log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    filename=f"../logs/{MICROSERVICE_NAME}.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(MICROSERVICE_NAME)
logger.info("Starting XGBoost Up/Down Classifier training")

# ============================================================================
# ClickHouse Connection
# ============================================================================
logger.info("Connecting to ClickHouse database")
try:
    client = Client(host='localhost')
    logger.info("Connected to ClickHouse successfully")
except Exception as e:
    logger.error(f"Failed to connect to ClickHouse: {str(e)}", exc_info=True)
    raise

# ============================================================================
# Data Extraction from ClickHouse
# ============================================================================
logger.info("Querying stock technical indicators from ClickHouse")
query_stock = """
    SELECT
        symbol, datetime, close, sigma_forecast, arma_forecast, ema_trend_filter_trend_up, ema_trend_filter_trend_down, long_term_bias_trend_up, long_term_bias_trend_down,
        macd_signal, risk_adj_ret, long_signal, short_signal, rsi_timing, weighted_avg_sentiment_api, weighted_avg_sentiment_llm, hour, day_of_week, day_of_month,
        month, quarter, year, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, ret_1, ret_2, ret_3, ret_6, vol_10
    FROM results_temp_table
    ORDER BY ts_ms ASC, symbol
"""

try:
    data_stock = client.execute("USE market_data")
    data_stock = client.execute(query_stock)
    logger.info(f"Retrieved {len(data_stock)} stock records")
except Exception as e:
    logger.error(f"Failed to query stock data: {str(e)}", exc_info=True)
    raise

columns_stock = [
    'symbol', 'datetime', 'close', 'sigma_forecast', 'arma_forecast', 'ema_trend_filter_trend_up', 'ema_trend_filter_trend_down',
    'long_term_bias_trend_up', 'long_term_bias_trend_down',
    'macd_signal', 'risk_adj_ret', 'long_signal', 'short_signal', 'rsi_timing', 'weighted_avg_sentiment_api',
    'weighted_avg_sentiment_llm', 'hour', 'day_of_week', 'day_of_month',
    'month', 'quarter', 'year', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'ret_1', 'ret_2', 'ret_3', 'ret_6', 'vol_10'
]

df = pd.DataFrame(data_stock, columns=columns_stock)
logger.info(f"Created DataFrame: {len(df)} rows")

df['datetime'] = pd.to_datetime(df['datetime'])
df['symbol']   = df['symbol'].astype(str)

numeric_cols = [c for c in df.columns if c not in ('symbol', 'datetime')]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================================
# XGBoost Training
# ============================================================================
print("=" * 80)
print("XGBoost Training")
print("=" * 80)

# 1. Sort data
df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
print(f"\n1. Loaded {len(df):,} rows")
print(f"   Symbols: {df['symbol'].unique()}")
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# 2. Data quality check + leakage-free target construction
print("\n2. Computing forward return from close prices (leakage-free target)...")

# The model predicts whether price will be HIGHER or LOWER after N=12 bars.
# We use close prices directly to compute this — no pct_change from the DB.
#
# For each row at time T:
#   future_close  = close at T+N  (shift(-N) per symbol)
#   forward_ret   = (future_close - close) / close * 100
#   up_down       = 1 if forward_ret > 0 else 0
#
# Why use close shift(-N) instead of summing N individual bar returns?
#   → Summing shifted pct_changes from the DB carries the same leakage risk
#     as using pct_change directly (each bar's return is computed in the DB
#     from that bar's own close, so the sum telescopes to the same quantity).
#   → A single shift(-N) on close is mathematically equivalent and cleaner.
#
# 'close' is dropped from features after target construction — the model
# must not see the current price level, only derived signals.

N = 12  # predict direction over next 60 minutes (12 x 5-min bars)

df['future_close'] = df.groupby('symbol')['close'].shift(-N)
df['forward_ret']  = (df['future_close'] - df['close']) / df['close'] * 100

nan_target = df['forward_ret'].isna().sum()
print(f"   NaN in forward_ret (last {N} rows per symbol, expected): {nan_target:,}")

df['forward_ret'] = pd.to_numeric(df['forward_ret'], errors='coerce')

inf_target = np.isinf(df['forward_ret'].to_numpy()).sum()
print(f"   Inf in forward_ret: {inf_target:,}")

# Drop rows where target cannot be computed (last N candles per symbol)
df = df.dropna(subset=['forward_ret'])
df = df[~np.isinf(df['forward_ret'])]
print(f"   Rows after dropping untargetable rows: {len(df):,}")

# Drop helper and raw price columns — neither should be a feature
df = df.drop(columns=['future_close', 'close'])

# 3. Split by date
print("\n3. Splitting data by date...")
train_df = df[(df['datetime'] >= '2021-01-04') & (df['datetime'] <= '2023-12-29')].copy()
test_df  = df[(df['datetime'] >= '2024-01-02') & (df['datetime'] <= '2024-12-30')].copy()

train_df = train_df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
test_df  = test_df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

print(f"   Training: {len(train_df):,} rows ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
print(f"   Test:     {len(test_df):,} rows ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

# 4. Create directional binary target from forward return
print("\n4. Creating binary target (up=1 if forward_ret > 0)...")
train_df['up_down'] = (train_df['forward_ret'] > 0).astype(int)
test_df['up_down']  = (test_df['forward_ret']  > 0).astype(int)

print(f"   Train target distribution:\n{train_df['up_down'].value_counts(normalize=True).to_string()}")
print(f"   Test  target distribution:\n{test_df['up_down'].value_counts(normalize=True).to_string()}")

print(train_df[['symbol', 'datetime', 'forward_ret', 'up_down']].head(20))

# 5. Symbol mapping
print("\n5. Creating symbol mapping...")
try:
    with open("symbol_mapping.pkl", "rb") as f:
        symbol_mapping = pickle.load(f)
    print("   Loaded existing mapping")
except FileNotFoundError:
    unique_symbols = df['symbol'].unique()
    symbol_mapping = {i: sym for i, sym in enumerate(unique_symbols)}
    with open("symbol_mapping.pkl", "wb") as f:
        pickle.dump(symbol_mapping, f)
    print(f"   Created new mapping for {len(unique_symbols)} symbols")

reverse_mapping = {v: k for k, v in symbol_mapping.items()}

# 6. Map symbols to codes
print("\n6. Mapping symbols to codes...")
train_df['symbol'] = train_df['symbol'].map(reverse_mapping).astype(int)
test_df['symbol']  = test_df['symbol'].map(reverse_mapping).astype(int)

# 7. Prepare features and target
print("\n7. Preparing features and target...")

# # Calculating news surprise columns
# window = 89
#
# train_rolling_mean = (
#     train_df.groupby('symbol')['weighted_avg_sentiment_api']
#       .transform(
#           lambda x: x.shift(1).rolling(window, min_periods=20).mean()
#       )
# )
#
# train_rolling_std = (
#     train_df.groupby('symbol')['weighted_avg_sentiment_api']
#       .transform(
#           lambda x: x.shift(1).rolling(window, min_periods=20).std()
#       )
# )
#
# train_df['sentiment_zscore'] = (
#     (train_df['weighted_avg_sentiment_api'] - train_rolling_mean)
#     / (train_rolling_std + 1e-8)
# )
#
# test_rolling_mean = (
#     test_df.groupby('symbol')['weighted_avg_sentiment_api']
#       .transform(
#           lambda x: x.shift(1).rolling(window, min_periods=20).mean()
#       )
# )
#
# test_rolling_std = (
#     test_df.groupby('symbol')['weighted_avg_sentiment_api']
#       .transform(
#           lambda x: x.shift(1).rolling(window, min_periods=20).std()
#       )
# )
#
# test_df['sentiment_zscore'] = (
#     (test_df['weighted_avg_sentiment_api'] - test_rolling_mean)
#     / (test_rolling_std + 1e-8)
# )

X_train_full = train_df.drop(columns=['forward_ret', 'up_down'])
y_train_full = train_df['up_down']

# CRITICAL: Use time-based split for validation (not random!)
split_date = X_train_full['datetime'].quantile(0.9)

X_train = X_train_full[X_train_full['datetime'] < split_date]
y_train = y_train_full[X_train_full['datetime'] < split_date]

X_val = X_train_full[X_train_full['datetime'] >= split_date]
y_val = y_train_full[X_train_full['datetime'] >= split_date]

X_train = X_train.drop(columns=['datetime'])
X_val   = X_val.drop(columns=['datetime'])

print(f"   Training:   {len(X_train):,} samples")
print(f"   Validation: {len(X_val):,} samples")

# 8. Final data cleaning
print("\n8. Final data cleaning...")
print(f"   Before: Train={len(X_train):,}, Val={len(X_val):,}")

print(f"   NaN in y_train: {y_train.isna().sum()}")
print(f"   Inf in y_train: {np.isinf(y_train).sum()}")
print(f"   NaN in X_train: {X_train.isna().sum().sum()}")
print(f"   Inf in X_train: {np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()}")

valid_train = ~(y_train.isna() | np.isinf(y_train))
X_train = X_train[valid_train]
y_train = y_train[valid_train]

valid_val = ~(y_val.isna() | np.isinf(y_val))
X_val = X_val[valid_val]
y_val = y_val[valid_val]

train_clean = ~(X_train.isna().any(axis=1) |
                np.isinf(X_train.select_dtypes(include=[np.number])).any(axis=1))
X_train = X_train[train_clean]
y_train = y_train[train_clean]

val_clean = ~(X_val.isna().any(axis=1) |
              np.isinf(X_val.select_dtypes(include=[np.number])).any(axis=1))
X_val = X_val[val_clean]
y_val = y_val[val_clean]

print(y_train.value_counts(normalize=True))
print(f"   After: Train={len(X_train):,}, Val={len(X_val):,}")

# 9. Target distribution diagnostic
print("\n9. Target Variable Distribution:")
print(f"   Training mean: {y_train.mean():.4f}")
print(f"   Training std:  {y_train.std():.4f}")
print(f"   Training min:  {y_train.min():.4f}")
print(f"   Training max:  {y_train.max():.4f}")

# 10. Train model
print("\n10. Training XGBoost...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=300,
    learning_rate=0.01,
    max_depth=4,
    # early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ============================================================================
# Test Set Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("11. TEST SET EVALUATION")
print("=" * 80)

# Prepare test features
X_test = test_df.drop(columns=['forward_ret', 'up_down', 'datetime'])
y_test = test_df['up_down']

# Remove NaN / Inf rows
test_clean = ~(
    X_test.isna().any(axis=1) |
    np.isinf(X_test.select_dtypes(include=[np.number])).any(axis=1)
)

X_test = X_test[test_clean]
y_test = y_test[test_clean]

print(f"\nTest samples: {len(X_test):,}")

# Predictions
y_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred)
auc = roc_auc_score(y_test,y_prob)

print("\nClassification Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Feature Importances:")
print(importance_df.head(15))

model.save_model("xgb_dir_model.json")
print("\n✓ Model saved!")