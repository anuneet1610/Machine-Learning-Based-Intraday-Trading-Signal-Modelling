"""
Decision Service - XGBoost Big Move Classifier Training

This script trains an XGBoost binary classifier to predict significant stock price
movements (>0.5% change). It uses technical indicators from stock calculations and
sentiment scores from news analysis.

Input: ClickHouse database (final_table and sentiment_stream)
Output: Trained XGBoost model (xgb_classifier_model.json)
"""
from clickhouse_driver import Client
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
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
# Query stock technical indicators from the most recent 1000 timestamps
# This provides a diverse dataset across multiple stocks and time periods
logger.info("Querying stock technical indicators from ClickHouse")
query_stock = """
    SELECT symbol, timestamp, ts_ms, close, sigma_forecast, arma_forecast,
           ema_trend_filter_trend_up, ema_trend_filter_trend_down,
           long_term_bias_trend_up, long_term_bias_trend_down,
           macd_signal, risk_adj_ret, long_signal, short_signal,
           rsi_timing, pct_change
    FROM final_table
    WHERE ts_ms IN (
        SELECT DISTINCT ts_ms 
        FROM final_table 
        ORDER BY ts_ms DESC 
        LIMIT 1000
    )
    ORDER BY ts_ms DESC, symbol
"""

# Query news sentiment data from the latest cycle
# This provides sentiment context for each stock
logger.info("Querying news sentiment data from ClickHouse")
query_news = """
    SELECT symbol, news_titles, sentiment_scores, relevance_scores, weighted_avg_sentiment
    FROM market_data.sentiment_stream
    WHERE cycle = (SELECT MAX(cycle) FROM market_data.sentiment_stream)
    ORDER BY symbol
"""

try:
    data_stock = client.execute(query_stock)
    logger.info(f"Retrieved {len(data_stock)} stock records")
except Exception as e:
    logger.error(f"Failed to query stock data: {str(e)}", exc_info=True)
    raise
columns_stock = ['symbol','timestamp','ts_ms','close','sigma_forecast','arma_forecast',
           'ema_trend_filter_up','ema_trend_filter_down',
           'long_term_bias_trend_up','long_term_bias_trend_down',
           'macd_signal','risk_adj_ret','long_signal','short_signal',
           'rsi_timing','pct_change']

try:
    data_news = client.execute(query_news)
    logger.info(f"Retrieved {len(data_news)} news sentiment records")
except Exception as e:
    logger.error(f"Failed to query news data: {str(e)}", exc_info=True)
    raise

# ============================================================================
# Data Preparation
# ============================================================================
# Convert query results to DataFrames
columns_stock = ['symbol','timestamp','ts_ms','close','sigma_forecast','arma_forecast',
           'ema_trend_filter_up','ema_trend_filter_down',
           'long_term_bias_trend_up','long_term_bias_trend_down',
           'macd_signal','risk_adj_ret','long_signal','short_signal',
           'rsi_timing','pct_change']

columns_news = ['symbol', 'news_titles', 'sentiment_scores', 'relevance_scores', 'weighted_avg_sentiment']

df_stock = pd.DataFrame(data_stock, columns=columns_stock)
df_news = pd.DataFrame(data_news, columns=columns_news)
logger.info(f"Created DataFrames: stock={len(df_stock)} rows, news={len(df_news)} rows")

# Merge stock and news data on symbol (left join to keep all stock records)
cumm_df = df_stock.merge(
    df_news[['symbol', 'weighted_avg_sentiment']],
    on='symbol',
    how='left'  # Keep all stock records even if no news data
)
logger.info(f"Merged dataset: {len(cumm_df)} rows")

# Load symbol mapping (created during initial training)
# This maps stock symbols to integer codes for model training
symbol_mapping_file = "symbol_mapping.pkl"
if not Path(symbol_mapping_file).exists():
    logger.error(f"Symbol mapping file not found: {symbol_mapping_file}")
    raise FileNotFoundError(f"Symbol mapping file not found: {symbol_mapping_file}")

with open(symbol_mapping_file, "rb") as f:
    symbol_mapping = pickle.load(f)
logger.info(f"Loaded symbol mapping with {len(symbol_mapping)} symbols")
print("Loaded existing symbol mapping")

# Create reverse mapping (symbol -> code) for encoding
reverse_mapping = {v: k for k, v in symbol_mapping.items()}

# Map symbols to their integer codes
cumm_df['symbol'] = cumm_df['symbol'].map(reverse_mapping)
cumm_df['symbol'] = cumm_df['symbol'].astype(int)

# Drop non-numeric / unnecessary columns
cumm_df = cumm_df.drop(columns=['timestamp'])

# Create binary target: "big move" = 1 if price change > 0.5%, else 0
# This is a classification problem: predict significant price movements
cumm_df['big_move'] = (cumm_df['pct_change'] > 2).astype(int)
big_move_count = cumm_df['big_move'].sum()
logger.info(f"Target distribution: {big_move_count} big moves out of {len(cumm_df)} total ({big_move_count/len(cumm_df)*100:.2f}%)")

# Fill missing sentiment scores with 0 (neutral sentiment)
cumm_df['weighted_avg_sentiment'] = cumm_df['weighted_avg_sentiment'].fillna(0)

# ============================================================================
# Train-Test Split
# ============================================================================
# Split features and target
X = cumm_df.drop(columns=['pct_change', 'big_move'])
y = cumm_df['big_move']

# Split into train/test sets (80/20)
# shuffle=False to maintain temporal order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

# ============================================================================
# Model Training
# ============================================================================
logger.info("Training XGBoost binary classifier...")
print("Training...")

# Configure XGBoost classifier with hyperparameters tuned for imbalanced data
# scale_pos_weight handles class imbalance (more non-big-moves than big-moves)
model = xgb.XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle class imbalance
    n_estimators=900,  # Number of boosting rounds
    learning_rate=0.02,  # Shrinkage parameter (lower = more conservative)
    max_depth=7,  # Maximum tree depth
    subsample=0.8,  # Row sampling ratio
    colsample_bytree=0.8,  # Column sampling ratio
)

try:
    model.fit(X_train, y_train)
    logger.info("Model training completed successfully")
except Exception as e:
    logger.error(f"Model training failed: {str(e)}", exc_info=True)
    raise

# ============================================================================
# Model Evaluation
# ============================================================================
from sklearn.metrics import precision_recall_curve

# Get prediction probabilities for precision-recall curve analysis
y_scores = model.predict_proba(X_test)[:, 1]  # Probability of positive class
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
logger.debug(f"Precision-Recall curve calculated with {len(thresholds)} thresholds")

# Generate predictions and evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

logger.info("=" * 60)
logger.info("Model Evaluation Results:")
logger.info(f"Accuracy: {accuracy:.4f}")
logger.info(f"\nClassification Report:\n{classification_report(y_test, preds)}")
logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, preds)}")
logger.info("=" * 60)

print("Accuracy:", accuracy)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

# ============================================================================
# Save Model
# ============================================================================
model_file = "xgb_classifier_model.json"
try:
    model.save_model(model_file)
    logger.info(f"Model saved successfully to {model_file}")
except Exception as e:
    logger.error(f"Failed to save model: {str(e)}", exc_info=True)
    raise

logger.info("Big Move Classifier training completed successfully")