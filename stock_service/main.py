"""
Stock Service - Historical Data Replay and Streaming Microservice

This service replays historical stock market data from CSV files and streams it
to Kafka in real-time (with speedup factor). This simulates live market data
for testing and development purposes.

Input: CSV file with historical stock data (merged_stock_popular_red.csv)
Output: Kafka topic 'stock_table' with real-time streaming stock quotes
"""
import pathway as pw
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Load environment variables
load_dotenv()

# Service configuration
MICROSERVICE_NAME = "stock_service"

# Ensure logs directory exists
log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging for production
# Using 'a' (append) mode instead of 'w' to preserve logs across restarts
logging.basicConfig(
    level=logging.INFO,
    filename=f"../logs/{MICROSERVICE_NAME}.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(MICROSERVICE_NAME)
logger.info(f"Starting {MICROSERVICE_NAME} - Initializing stock data replay pipeline")

# ============================================================================
# Input Schema Definition
# ============================================================================
class InputSchema(pw.Schema):
    """
    Schema for historical stock quote data from CSV file.
    
    Attributes:
        timestamp: Human-readable timestamp string
        symbol: Stock ticker symbol (e.g., 'AAPL')
        open: Opening price for the period
        high: Highest price during the period
        low: Lowest price during the period
        close: Closing price for the period
        volume: Trading volume (number of shares)
        ts_ms: Timestamp in milliseconds (used for temporal replay)
    """
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    ts_ms: int

# ============================================================================
# Historical Data Replay
# ============================================================================
# Replay CSV file with time-based simulation
# This reads historical data and replays it in real-time with a speedup factor
# Speedup of 30.0 means data is replayed 30x faster than real-time
csv_file = "merged_stock_popular_red.csv"
if not Path(csv_file).exists():
    logger.error(f"CSV file not found: {csv_file}")
    raise FileNotFoundError(f"CSV file not found: {csv_file}")

logger.info(f"Reading historical data from {csv_file}")
logger.info("Replay configuration: speedup=30.0 (30x faster than real-time)")

try:
    quotes = pw.demo.replay_csv_with_time(
        path=csv_file,
        schema=InputSchema,
        time_column="ts_ms",  # Column used for temporal ordering
        unit="ms",  # Time unit: milliseconds
        speedup=30.0  # Replay speed multiplier (30x faster)
    )
    logger.info("Historical data replay configured successfully")
except Exception as e:
    logger.error(f"Failed to configure data replay: {str(e)}", exc_info=True)
    raise

# ============================================================================
# Data Transformation
# ============================================================================
# Transform data into Kafka-compatible format
# Select and structure columns for downstream consumption
logger.info("Transforming data for Kafka output")

output = quotes.select(
    key=quotes.symbol,  # Partition key for Kafka
    value=pw.apply(
        lambda symbol, o, h, l, c, v, t, ts_ms: {
            "symbol": symbol,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "timestamp": t,
            "ts_ms": ts_ms
        },
        quotes.symbol,
        quotes.open,
        quotes.high,
        quotes.low,
        quotes.close,
        quotes.volume,
        quotes.timestamp,
        quotes.ts_ms
    ),
)



# ============================================================================
# Kafka Output Configuration
# ============================================================================
# Write streaming stock data to Kafka topic
kafka_broker = os.getenv("KAFKA_BROKER")
if not kafka_broker:
    logger.error("KAFKA_BROKER environment variable not set")
    raise ValueError("KAFKA_BROKER environment variable is required")

logger.info(f"Configuring Kafka writer for topic 'stock_table' (broker: {kafka_broker})")
try:
    pw.io.kafka.write(
        output,
        {
            "bootstrap.servers": kafka_broker,
            "group.id": "pathway-group",
        },
        topic_name="stock_table",
        format="json",  # JSON format for structured data
    )
    logger.info("Kafka writer configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Kafka writer: {str(e)}", exc_info=True)
    raise

# ============================================================================
# Start Pathway Pipeline
# ============================================================================
# Run the Pathway computation graph
# This will start replaying historical data and streaming to Kafka
logger.info("Starting Pathway computation pipeline...")
logger.info("Streaming stock data to Kafka topic 'stock_table'")
logger.info("Press Ctrl+C to stop")
try:
    pw.run()
except KeyboardInterrupt:
    logger.info("Pipeline interrupted by user")
    raise
except Exception as e:
    logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
    raise
