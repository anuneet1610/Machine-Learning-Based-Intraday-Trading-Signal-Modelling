"""
Decision Service - XGBoost Percentage Change Regressor Training

This script trains an XGBoost regression model to predict stock price percentage
changes. It uses technical indicators from stock calculations and sentiment scores
from news analysis.

Input: ClickHouse database (final_table and sentiment_stream)
Output: Trained XGBoost model (xgb_pct_change_model.json)
"""
from clickhouse_driver import Client
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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
logger.info("Starting XGBoost Percentage Change Regressor training")

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

columns_news = ['symbol', 'news_titles', 'sentiment_scores', 'relevance_scores', 'weighted_avg_sentiment']

# ============================================================================
# Data Preparation
# ============================================================================
df_stock = pd.DataFrame(data_stock, columns=columns_stock)
df_news = pd.DataFrame(data_news, columns=columns_news)
logger.info(f"Created DataFrames: stock={len(df_stock)} rows, news={len(df_news)} rows")

# Merge stock and news data
cumm_df = df_stock.merge(
    df_news[['symbol', 'weighted_avg_sentiment']],
    on='symbol',
    how='left'
)
logger.info(f"Merged dataset: {len(cumm_df)} rows")

# Load symbol mapping
symbol_mapping_file = "symbol_mapping.pkl"
if not Path(symbol_mapping_file).exists():
    logger.error(f"Symbol mapping file not found: {symbol_mapping_file}")
    raise FileNotFoundError(f"Symbol mapping file not found: {symbol_mapping_file}")

with open(symbol_mapping_file, "rb") as f:
    symbol_mapping = pickle.load(f)
logger.info(f"Loaded symbol mapping with {len(symbol_mapping)} symbols")
print("Loaded existing symbol mapping")

# Create reverse mapping and encode symbols
reverse_mapping = {v: k for k, v in symbol_mapping.items()}
cumm_df['symbol'] = cumm_df['symbol'].map(reverse_mapping)
cumm_df['symbol'] = cumm_df['symbol'].astype(int)

# Fill missing sentiment scores
cumm_df['weighted_avg_sentiment'] = cumm_df['weighted_avg_sentiment'].fillna(0)

# Drop non-numeric columns
cumm_df = cumm_df.drop(columns=['timestamp'])

# ============================================================================
# Train-Test Split
# ============================================================================
# Split features and target (pct_change is the target for regression)
X = cumm_df.drop(columns=['pct_change'])
y = cumm_df['pct_change']

logger.info(f"Target statistics: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

# ============================================================================
# Model Training
# ============================================================================
logger.info("Training XGBoost regression model...")

# Configure XGBoost regressor with regularization to prevent overfitting
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression with squared error loss
    n_estimators=800,  # Number of boosting rounds
    learning_rate=0.01,  # Shrinkage parameter
    max_depth=8,  # Maximum tree depth
    min_child_weight=3,  # Minimum sum of instance weight in child
    subsample=0.75,  # Row sampling ratio
    colsample_bytree=0.75,  # Column sampling ratio
    reg_lambda=1.0,  # L2 regularization
    reg_alpha=0.3,  # L1 regularization
    gamma=0.1  # Minimum loss reduction for split
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
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

logger.info("=" * 60)
logger.info("Model Evaluation Results:")
logger.info(f"Mean Squared Error (MSE): {mse:.6f}")
logger.info(f"R² Score: {r2:.4f}")
logger.info("=" * 60)

print("MSE:", mse)
print("R2 Score:", r2)

# ============================================================================
# Save Model
# ============================================================================
model_file = "xgb_pct_change_model.json"
try:
    model.save_model(model_file)
    logger.info(f"Model saved successfully to {model_file}")
except Exception as e:
    logger.error(f"Failed to save model: {str(e)}", exc_info=True)
    raise

logger.info("Percentage Change Regressor training completed successfully")