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
logger.info("Starting XGBoost Big Move Classifier training")

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
    client.execute("USE market_data")
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
print("\n2. Computing forward pct_change (leakage-free target)...")

# The model must predict the NEXT candle's move using CURRENT candle's features.
# We compute next-candle pct_change via a forward shift per symbol:
#   future_close  = close shifted -1 (the next candle's close, within the same symbol)
#   fwd_pct_change = (future_close - close) / close * 100
#
# This means: row at time T has features from T and label = move from T to T+1.
# The last row of each symbol will have NaN future_close → dropped.
# 'close' itself is dropped from features to avoid leakage of current price level.

df['future_close'] = df.groupby('symbol')['close'].shift(-1)
df['fwd_pct_change'] = (df['future_close'] - df['close']) / df['close'] * 100

nan_target = df['fwd_pct_change'].isna().sum()
print(f"   NaN in fwd_pct_change (last row per symbol, expected): {nan_target:,}")

inf_target = np.isinf(df['fwd_pct_change']).sum()
print(f"   Inf in fwd_pct_change: {inf_target:,}")

# Drop rows where target cannot be computed (last candle per symbol)
df = df.dropna(subset=['fwd_pct_change'])
df = df[~np.isinf(df['fwd_pct_change'])]
print(f"   Rows after dropping untargetable rows: {len(df):,}")

# Drop helper columns — neither close nor future_close should be a feature
# (close leaks current price level; future_close IS the target)
df = df.drop(columns=['future_close', 'close'])

# 3. Split by date
print("\n3. Splitting data by date...")
train_df = df[(df['datetime'] >= '2021-01-04') & (df['datetime'] <= '2023-12-29')].copy()
test_df  = df[(df['datetime'] >= '2024-01-02') & (df['datetime'] <= '2024-12-30')].copy()

print(f"   Training: {len(train_df):,} rows ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
print(f"   Test:     {len(test_df):,} rows ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

# 4. Create binary target from forward return (computed AFTER split to avoid any look-ahead)
print("\n4. Creating binary target (big move = |fwd_pct_change| > 0.2)...")
train_df['up_down'] = (abs(train_df['fwd_pct_change']) > 0.2).astype(int)
test_df['up_down']  = (abs(test_df['fwd_pct_change'])  > 0.2).astype(int)

print(f"   Train target distribution:\n{train_df['up_down'].value_counts(normalize=True).to_string()}")
print(f"   Test  target distribution:\n{test_df['up_down'].value_counts(normalize=True).to_string()}")

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
#
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

X_train_full = train_df.drop(columns=['fwd_pct_change', 'up_down'])
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
    n_estimators=800,
    learning_rate=0.01,
    max_depth=8,
    reg_lambda=1.0,
    reg_alpha=0.3,
    eval_metric='logloss',
    use_label_encoder=False,
    early_stopping_rounds=50
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
X_test = test_df.drop(columns=['fwd_pct_change', 'up_down', 'datetime'])
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
y_pred = (y_prob > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test,y_pred,)

recall = recall_score(
    y_test,
    y_pred,
)

f1 = f1_score(
    y_test,
    y_pred,
)

auc = roc_auc_score(
    y_test,
    y_prob,
)

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

model.save_model("xgb_big_move_model.json")
print("\n✓ Model saved!")