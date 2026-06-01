"""
Decision Service - Real-time Trading Alert Inference Service

This service consumes stock calculation data from Kafka topic 'metrics_data',
performs real-time predictions using two trained XGBoost models (magnitude +
direction), and sends alerts to Kafka topic 'alert' when a significant price
move is predicted in a clear direction.

Signal logic (mirrors xgboost_backtest.py):
  - mag_model  : predicts probability of a *big* price move  (prob_big)
  - dir_model  : predicts probability that the move is *up*  (prob_up)
  - trade_signal = signal * direction  where:
        signal    = 1  if prob_big > THRESHOLD (0.35)  else 0
        direction = +1 if prob_up  > 0.53
                   -1 if prob_up  < 0.42
                    0 otherwise  (no clear direction → no alert)

Alert is sent only when trade_signal != 0  (i.e. big move predicted AND
direction is clear).

Input : Kafka topic 'metrics_data'
Output: Kafka topic 'alert'  (rows where trade_signal != 0)
        ClickHouse table market_data.final_table  (every row)
"""

import json
import collections
import numpy as np
import pickle
import xgboost as xgb
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
from clickhouse_driver import Client
import logging
from datetime import datetime as dt

# ============================================================================
# Configuration  (mirrors backtest constants)
# ============================================================================
MICROSERVICE_NAME  = "decision_service"

MAG_MODEL_FILE     = "xgb_big_move_model.json"   # big-move classifier
DIR_MODEL_FILE     = "xgb_dir_model.json"                  # direction classifier
SYMBOL_MAPPING_FILE = "symbol_mapping.pkl"

THRESHOLD          = 0.35   # prob_big threshold to trigger a trade signal
DIR_LONG_CUT       = 0.53   # prob_up above this  → long  (+1)
DIR_SHORT_CUT      = 0.42   # prob_up below this  → short (-1)

KAFKA_BOOTSTRAP    = "localhost:9092"
INPUT_TOPIC        = "metrics_data"
OUTPUT_TOPIC       = "alert"
CONSUMER_GROUP     = "decision-service-group"

CLICKHOUSE_HOST    = "localhost"
CLICKHOUSE_DB      = "market_data"
CLICKHOUSE_TABLE   = "final_table"

# ============================================================================
# Logging
# ============================================================================
log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=f"../logs/{MICROSERVICE_NAME}.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(MICROSERVICE_NAME)
logger.info("Starting Decision Service — real-time inference pipeline")

# ============================================================================
# Model Loading
# ============================================================================
logger.info("Loading trained XGBoost models")

for path in (MAG_MODEL_FILE, DIR_MODEL_FILE):
    if not Path(path).exists():
        logger.error(f"Model file not found: {path}")
        raise FileNotFoundError(f"Model file not found: {path}")

try:
    mag_model = xgb.XGBClassifier()
    mag_model.load_model(MAG_MODEL_FILE)
    logger.info(f"Loaded magnitude model from {MAG_MODEL_FILE}")

    dir_model = xgb.XGBClassifier()
    dir_model.load_model(DIR_MODEL_FILE)
    logger.info(f"Loaded direction model from {DIR_MODEL_FILE}")
except Exception as e:
    logger.error(f"Failed to load models: {e}", exc_info=True)
    raise

# ============================================================================
# Symbol Mapping
# ============================================================================
if not Path(SYMBOL_MAPPING_FILE).exists():
    logger.error(f"Symbol mapping file not found: {SYMBOL_MAPPING_FILE}")
    raise FileNotFoundError(f"Symbol mapping file not found: {SYMBOL_MAPPING_FILE}")

with open(SYMBOL_MAPPING_FILE, "rb") as f:
    symbol_mapping = pickle.load(f)          # int -> symbol string  (as saved)

# Invert to  symbol_string -> int  for encoding incoming messages
symbol_to_code = {v: k for k, v in symbol_mapping.items()}
logger.info(f"Loaded symbol mapping with {len(symbol_mapping)} symbols")

# ============================================================================
# ClickHouse Connection & Table Bootstrap
# ============================================================================
logger.info(f"Connecting to ClickHouse (host={CLICKHOUSE_HOST}, db={CLICKHOUSE_DB})")
try:
    ch_client = Client(host=CLICKHOUSE_HOST)
    ch_client.execute(f"CREATE DATABASE IF NOT EXISTS {CLICKHOUSE_DB}")
    ch_client.execute(f"USE {CLICKHOUSE_DB}")
    logger.info("Connected to ClickHouse successfully")
except Exception as e:
    logger.error(f"Failed to connect to ClickHouse: {e}", exc_info=True)
    raise

# Create final_table if it does not already exist.
# Columns = all raw metrics from metrics_data
#           +  model outputs (prob_big, prob_up, signal, direction, trade_signal)
# Engine : MergeTree, partitioned by month, ordered by (symbol, datetime).
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DB}.{CLICKHOUSE_TABLE}
(
    -- identity / time
    symbol                      String,
    datetime                    DateTime,
    ts_ms                       Int64,

    -- raw price / forecast metrics from metrics_data
    close                       Float64,
    sigma_forecast              Float64,
    arma_forecast               Float64,

    -- trend filter flags
    ema_trend_filter_trend_up   UInt8,
    ema_trend_filter_trend_down UInt8,
    long_term_bias_trend_up     UInt8,
    long_term_bias_trend_down   UInt8,

    -- technical indicators
    macd_signal                 Float64,
    risk_adj_ret                Float64,
    long_signal                 UInt8,
    short_signal                UInt8,
    rsi_timing                  Float64,

    -- sentiment features
    weighted_avg_sentiment_api  Float64,
    weighted_avg_sentiment_distilbert Float64,

    -- calendar features
    hour                        UInt8,
    day_of_week                 UInt8,
    day_of_month                UInt8,
    month                       UInt8,
    quarter                     UInt8,
    year                        UInt16,

    -- cyclical encodings
    hour_sin                    Float64,
    hour_cos                    Float64,
    day_sin                     Float64,
    day_cos                     Float64,
    month_sin                   Float64,
    month_cos                   Float64,

    -- return / vol features
    ret_1                       Float64,
    ret_2                       Float64,
    ret_3                       Float64,
    ret_6                       Float64,
    vol_10                      Float64,

    -- model outputs
    prob_big                    Float64,   -- magnitude model: P(big move)
    prob_up                     Float64,   -- direction model: P(move is up)
    signal                      UInt8,     -- 1 if prob_big > THRESHOLD else 0
    direction                   Int8,      -- +1 long / -1 short / 0 flat
    trade_signal                Int8       -- signal * direction
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(datetime)
ORDER BY (symbol, datetime)
"""

try:
    ch_client.execute(CREATE_TABLE_SQL)
    logger.info(f"Ensured {CLICKHOUSE_DB}.{CLICKHOUSE_TABLE} exists")
except Exception as e:
    logger.error(f"Failed to create {CLICKHOUSE_TABLE}: {e}", exc_info=True)
    raise

def parse_datetime(value: str) -> dt:
    try:
        return dt.fromisoformat(value)   # handles both "2024-01-15 10:30:00" and "2024-01-15T10:30:00"
    except (ValueError, TypeError):
        return dt.utcnow()

# ============================================================================
# Kafka — Consumer & Producer
# ============================================================================
logger.info(f"Configuring Kafka consumer for topic '{INPUT_TOPIC}'")
try:
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    logger.info("Kafka consumer configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Kafka consumer: {e}", exc_info=True)
    raise

logger.info(f"Configuring Kafka producer for topic '{OUTPUT_TOPIC}'")
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    logger.info("Kafka producer configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Kafka producer: {e}", exc_info=True)
    raise

logger.info(f"Waiting for messages on '{INPUT_TOPIC}'...")
print(f"Waiting for messages on '{INPUT_TOPIC}'...")

# ============================================================================
# Main Processing Loop
# ============================================================================
logger.info("Starting main processing loop")

try:
    for msg in consumer:
        data = msg.value
        logger.debug(f"Received message: {json.dumps(data, indent=2)}")
        print("Received:", data)

        # ------------------------------------------------------------------
        # Validate symbol
        # ------------------------------------------------------------------
        data["symbol"] = data["symbol"].strip()
        current_symbol = data["symbol"]

        if not current_symbol:
            logger.warning("Received message with empty symbol, skipping")
            continue

        symbol_encoded = symbol_to_code.get(current_symbol, -1)
        if symbol_encoded == -1:
            logger.warning(f"Unknown symbol '{current_symbol}', not in training data — skipping")
            print(f"Unknown symbol: {current_symbol}, skipping")
            continue

        # ------------------------------------------------------------------
        # Validate sentiment fields — treat None/missing as 0.0
        # ------------------------------------------------------------------


        # ------------------------------------------------------------------
        # Build feature vector  (same column order as training)
        # ------------------------------------------------------------------
        try:
            features = np.array([[
                symbol_encoded,
                # float(data["close"]),
                # int(data.get("ts_ms")),
                float(data["sigma_forecast"]),
                float(data["arma_forecast"]),
                int(data["ema_trend_filter_trend_up"]),
                int(data["ema_trend_filter_trend_down"]),
                int(data["long_term_bias_trend_up"]),
                int(data["long_term_bias_trend_down"]),
                float(data["macd_signal"]),
                float(data["risk_adj_ret"]),
                int(data["long_signal"]),
                int(data["short_signal"]),
                float(data["rsi_timing"]),
                float(data["weighted_avg_sentiment_api"] or 0.0),
                float(data["weighted_avg_sentiment_distilbert"]),
                int(data["hour"]),
                int(data["day_of_week"]),
                int(data["day_of_month"]),
                int(data["month"]),
                int(data["quarter"]),
                int(data["year"]),
                float(data["hour_sin"]),
                float(data["hour_cos"]),
                float(data["day_sin"]),
                float(data["day_cos"]),
                float(data["month_sin"]),
                float(data["month_cos"]),
                float(data["ret_1"]),
                float(data["ret_2"]),
                float(data["ret_3"]),
                float(data["ret_6"]),
                float(data["vol_10"]),
            ]])
        except KeyError as e:
            logger.error(f"Missing required field for {current_symbol}: {e}")
            continue
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data type for {current_symbol}: {e}")
            continue

        # ------------------------------------------------------------------
        # Model Predictions  (mirrors backtest signal generation)
        # ------------------------------------------------------------------
        try:
            prob_big = float(mag_model.predict_proba(features)[:, 1][0])
            prob_up  = float(dir_model.predict_proba(features)[:, 1][0])
        except Exception as e:
            logger.error(f"Prediction failed for {current_symbol}: {e}", exc_info=True)
            continue

        # Signal logic — identical to backtest lines 229-232
        signal    = 1 if prob_big > THRESHOLD else 0
        if   prob_up > DIR_LONG_CUT:
            direction = 1
        elif prob_up < DIR_SHORT_CUT:
            direction = -1
        else:
            direction = 0

        trade_signal = signal * direction  # +1 long, -1 short, 0 flat

        logger.info(
            f"{current_symbol}: prob_big={prob_big:.4f} prob_up={prob_up:.4f} "
            f"signal={signal} direction={direction} trade_signal={trade_signal}"
        )
        print(
            f"{current_symbol}: prob_big={prob_big:.4f}  prob_up={prob_up:.4f}  "
            f"trade_signal={trade_signal:+d}"
        )

        # ------------------------------------------------------------------
        # Persist every row to ClickHouse  (raw metrics + model outputs)
        # ------------------------------------------------------------------
        try:
            ch_row = [(
                current_symbol,
                parse_datetime(data.get("datetime", "")),
                int(data.get("ts_ms")),
                float(data["close"]),
                float(data["sigma_forecast"]),
                float(data["arma_forecast"]),
                int(data["ema_trend_filter_trend_up"]),
                int(data["ema_trend_filter_trend_down"]),
                int(data["long_term_bias_trend_up"]),
                int(data["long_term_bias_trend_down"]),
                float(data["macd_signal"]),
                float(data["risk_adj_ret"]),
                int(data["long_signal"]),
                int(data["short_signal"]),
                float(data["rsi_timing"]),
                float(data["weighted_avg_sentiment_api"] or 0.0),
                float(data["weighted_avg_sentiment_distilbert"]),
                int(data["hour"]),
                int(data["day_of_week"]),
                int(data["day_of_month"]),
                int(data["month"]),
                int(data["quarter"]),
                int(data["year"]),
                float(data["hour_sin"]),
                float(data["hour_cos"]),
                float(data["day_sin"]),
                float(data["day_cos"]),
                float(data["month_sin"]),
                float(data["month_cos"]),
                float(data["ret_1"]),
                float(data["ret_2"]),
                float(data["ret_3"]),
                float(data["ret_6"]),
                float(data["vol_10"]),
                prob_big,
                prob_up,
                signal,
                direction,
                trade_signal,
            )]
            ch_client.execute(
                f"INSERT INTO {CLICKHOUSE_DB}.{CLICKHOUSE_TABLE} VALUES",
                ch_row
            )
            logger.debug(f"ClickHouse row inserted for {current_symbol}")
        except Exception as e:
            logger.error(f"ClickHouse insert failed for {current_symbol}: {e}", exc_info=True)

        print("Calculated: ", ch_row)

        # ------------------------------------------------------------------
        # Alert Generation — only when trade_signal != 0
        # ------------------------------------------------------------------
        if trade_signal != 0:
            direction_label = "LONG" if trade_signal == 1 else "SHORT"

            record = {
                "symbol"                  : current_symbol,
                "direction"               : direction_label,
                "trade_signal"            : trade_signal,           # +1 or -1
                "prob_big"                : prob_big,               # magnitude model output
                "prob_up"                 : prob_up,                # direction model output
                "close"                   : float(data["close"]),
                "sigma_forecast"          : float(data["sigma_forecast"]),
                "ema_filter_trend_up"     : int(data.get("ema_trend_filter_trend_up", 0)),
                "ema_filter_trend_down"   : int(data.get("ema_trend_filter_trend_down", 0)),
            }

            try:
                producer.send(OUTPUT_TOPIC, value=record)
                producer.flush()
                logger.info(
                    f"Alert sent for {current_symbol}: {direction_label}  "
                    f"prob_big={prob_big:.4f}  prob_up={prob_up:.4f}"
                )
                print(f"Alert sent for {current_symbol}: {direction_label}")
            except Exception as e:
                logger.error(f"Failed to send alert for {current_symbol}: {e}", exc_info=True)

except KeyboardInterrupt:
    logger.info("Processing interrupted by user")
except Exception as e:
    logger.error(f"Fatal error in processing loop: {e}", exc_info=True)
    raise
finally:
    if consumer:
        consumer.close()
        logger.info("Kafka consumer closed")
    if producer:
        producer.close()
        logger.info("Kafka producer closed")
    if ch_client:
        ch_client.disconnect()
        logger.info("ClickHouse connection closed")
