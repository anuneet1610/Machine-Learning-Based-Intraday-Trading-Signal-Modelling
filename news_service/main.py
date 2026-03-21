"""
News Service - Financial News Ingestion and Sentiment Analysis Microservice

This service fetches financial news from Alpha Vantage API, performs sentiment
analysis, aggregates news by stock ticker, and publishes results to Kafka and
ClickHouse. It dynamically adjusts date ranges based on timestamps from the
stock service.

Input: Alpha Vantage News API, Kafka topic 'stock_timestamp' for date ranges
Output: Kafka topic 'News' with aggregated sentiment data, ClickHouse database
"""
import json
import time
import hashlib
import requests
from dotenv import load_dotenv
from kafka import KafkaProducer, KafkaConsumer
from datetime import datetime, timedelta
from threading import Thread, Lock
import os
from pathlib import Path
import logging

# Load environment variables
load_dotenv()

# Service configuration
MICROSERVICE_NAME = "news_service"

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
logger.info(f"Starting {MICROSERVICE_NAME} - Initializing news ingestion pipeline")

ALPHA_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_URL")
KAFKA_TOPIC = "News"
TIMESTAMP_TOPIC = "stock_timestamp"

top_240_popular_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "INTC",
    "QCOM", "AVGO", "TXN", "MU", "LRCX", "AMAT", "ASML", "TSM", "ADBE", "CRM",
    "ORCL", "CSCO", "IBM", "NOW", "SHOP", "SNOW", "UBER", "ABNB", "DASH", "COIN",
    "RIVN", "LCID", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR", "JPM", "BAC",
    "WFC", "GS", "MS", "C", "SCHW", "BLK", "PYPL", "SQ", "SOFI", "HOOD",
    "MA", "V", "AXP", "KO", "PEP", "PG", "JNJ", "MRK", "PFE", "LLY",
    "ABBV", "BMY", "GILD", "AMGN", "REGN", "VRTX", "MRNA", "BNTX", "XOM", "CVX",
    "COP", "SLB", "HAL", "OXY", "MPC", "PSX", "FCX", "NEM", "GOLD", "CAT",
    "DE", "GE", "UNP", "CSX", "LMT", "RTX", "BA", "HON", "MMM", "GM",
    "F", "NIO", "XPEV", "LI", "BIDU", "BABA", "JD", "PDD", "NTES", "PLTR",
    "SNAP", "PINS", "ROKU", "ZM", "DOCU", "TWLO", "CRWD", "ZS", "NET", "PANW",
    "FTNT", "OKTA", "DDOG", "MDB", "HUBS", "TEAM", "TTD", "DKNG", "RBLX", "U",
    "PATH", "GTLB", "AFRM", "UPST", "TOST", "RKLB", "IONQ", "QBTS", "SOUN", "AI",
    "SMCI", "ARM", "MSTR", "MARA", "CLSK", "RIOT", "CIFR", "HUT", "BITF", "WULF",
    "IREN", "CORZ", "BTBT", "CAN", "SOS", "AMC", "GME", "BB", "KOSS", "EXPR",
    "SPCE", "RIDE", "NKLA", "LCID", "RIVN", "FFIE", "MULN", "HYLN", "GOEV", "PSNY",
    "WISH", "CLSK", "HOOD", "UPST", "OPEN", "AI", "BBAI", "SOUN", "BIGC", "LAZR",
    "INDI", "OUST", "AEVA", "INVZ", "DNA", "PACB", "ILMN", "TWST", "BEAM", "CRSP",
    "NTLA", "EDIT", "VIR", "ARCT", "ABCL", "EXAI", "SDGR", "RXRX", "RXRX", "RBLX",
    "UNH", "CI", "ELV", "HUM", "CVS", "WBA", "CNC", "MOH", "HCA", "UHS",
    "DVA", "THC", "EHC", "ACHC", "SEM", "BKD", "BHC", "TEVA", "VTRS", "ZTS",
    "BMY", "PFE", "MRK", "ABBV", "JNJ", "LLY", "NVO", "AZN", "SNY", "GSK",
    "RHHBY", "NVS", "TAK", "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "WBA",
    "WMT", "COST", "TGT", "HD", "LOW", "DLTR", "DG", "ORLY", "AZO", "TSCO"
]

TOPICS = [
    "technology",
    "finance",
    "life_sciences",
    "blockchain",
    "mergers_and_acquisitions"
]


class TimestampManager:
    """
    Manages date range for news API queries by consuming timestamps from Kafka.
    
    This class maintains a sliding 10-day window based on the latest timestamp
    received from the stock service. It runs in a background thread to continuously
    update the date range as new timestamps arrive.
    """

    def __init__(self):
        """Initialize timestamp manager with default state."""
        self.current_end_date = None
        self.current_start_date = None
        self.lock = Lock()  # Thread-safe access to date range
        self.consumer = None
        self.running = False
        logger.debug("TimestampManager initialized")

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp string to datetime object.
        
        Tries multiple common timestamp formats to handle various input formats.
        
        Args:
            timestamp_str: Timestamp string in various formats
        
        Returns:
            datetime object if parsing succeeds, None otherwise
        """
        try:
            # Try common timestamp formats in order of likelihood
            formats = [
                "%Y-%m-%d %H:%M:%S",  # Standard format: 2024-01-15 14:30:00
                "%Y-%m-%d",  # Date only: 2024-01-15
                "%Y%m%d",  # Compact format: 20240115
                "%Y-%m-%dT%H:%M:%S"  # ISO-like format: 2024-01-15T14:30:00
            ]
            
            for fmt in formats:
                try:
                    parsed = datetime.strptime(timestamp_str, fmt)
                    logger.debug(f"Parsed timestamp '{timestamp_str}' using format '{fmt}'")
                    return parsed
                except ValueError:
                    continue
            
            # If all standard formats fail, try ISO format
            parsed = datetime.fromisoformat(timestamp_str)
            logger.debug(f"Parsed timestamp '{timestamp_str}' using ISO format")
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse timestamp '{timestamp_str}': {str(e)}")
            print(f"Failed to parse timestamp: {timestamp_str}, error: {e}")
            return None

    def _format_date_for_api(self, dt: datetime) -> str:
        """
        Format datetime to YYYYMMDDTHHMM format required by Alpha Vantage API.
        
        Args:
            dt: datetime object to format
        
        Returns:
            Formatted string: YYYYMMDDTHHMM (e.g., "20240115T1430")
        """
        return dt.strftime("%Y%m%dT%H%M")

    def start_consuming(self):
        """
        Start consuming timestamps from Kafka in a background thread.
        
        This method initializes a Kafka consumer and starts a daemon thread
        that continuously updates the date range based on received timestamps.
        """
        self.running = True
        kafka_broker = os.getenv("KAFKA_BROKER")
        if not kafka_broker:
            logger.error("KAFKA_BROKER environment variable not set")
            raise ValueError("KAFKA_BROKER environment variable is required")
        
        logger.info(f"Initializing Kafka consumer for topic '{TIMESTAMP_TOPIC}' (broker: {kafka_broker})")
        self.consumer = KafkaConsumer(
            TIMESTAMP_TOPIC,
            bootstrap_servers=kafka_broker,
            auto_offset_reset="latest",  # Start from latest messages (don't replay old timestamps)
            enable_auto_commit=True,
            group_id="news-timestamp-consumer",
            value_deserializer=lambda m: m.decode('utf-8')  # Decode bytes to string
        )

        thread = Thread(target=self._consume_loop, daemon=True)
        thread.start()
        logger.info("Started timestamp consumer thread (daemon)")
        print("Started timestamp consumer thread")


    def _consume_loop(self):
        """
        Continuously consume timestamps and update date range.
        
        This method runs in a background thread and polls Kafka for new timestamps.
        When a timestamp is received, it updates the date range to a 10-day window
        ending at the received timestamp.
        """
        logger.info("Timestamp consumer loop started")
        while self.running:
            try:
                # Poll for messages with 1 second timeout
                messages = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        timestamp_str = record.value
                        logger.debug(f"Received timestamp: {timestamp_str}")
                        dt = self._parse_timestamp(timestamp_str)

                        if dt:
                            with self.lock:
                                # Format dates for Alpha Vantage API (YYYYMMDDTHHMM)
                                # Use 10-day lookback window for news queries
                                self.current_end_date = self._format_date_for_api(dt)
                                start_dt = dt - timedelta(days=10)
                                self.current_start_date = self._format_date_for_api(start_dt)
                                logger.info(
                                    f"Updated date range: {self.current_start_date} to {self.current_end_date} "
                                    f"(10-day window)"
                                )
            except Exception as e:
                logger.error(f"Error in timestamp consumer: {str(e)}", exc_info=True)
                print(f"Error in timestamp consumer: {e}")
                time.sleep(1)  # Wait before retrying

    def get_date_range(self):
        """
        Get current start and end dates in a thread-safe manner.
        
        Returns:
            Tuple of (start_date, end_date) strings in YYYYMMDDTHHMM format,
            or (None, None) if no dates have been set yet.
        """
        with self.lock:
            return self.current_start_date, self.current_end_date

    def stop(self):
        """
        Stop the timestamp consumer gracefully.
        
        Sets running flag to False and closes Kafka consumer connection.
        """
        logger.info("Stopping timestamp consumer")
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Timestamp consumer closed")


class ClickHouseNewsWriter:
    """
    Handles writing aggregated news sentiment data to ClickHouse database.
    
    Uses HTTP interface for inserting data, which is simpler than native protocol
    for JSON data insertion.
    """
    
    def __init__(self):
        """Initialize ClickHouse writer and test connection."""
        self.url = os.getenv("CLICKHOUSE_URL")
        if not self.url:
            logger.error("CLICKHOUSE_URL environment variable not set")
            raise ValueError("CLICKHOUSE_URL environment variable is required")
        
        # Test connection to ClickHouse
        logger.info(f"Testing ClickHouse connection: {self.url}")
        try:
            response = requests.get(self.url, params={"query": "SELECT 1"}, timeout=5)
            if response.status_code == 200:
                logger.info("ClickHouse HTTP connection established successfully")
                print("ClickHouse HTTP connection established successfully")
            else:
                logger.warning(f"ClickHouse connection returned status {response.status_code}")
                print(f"ClickHouse connection warning: {response.status_code}")

        except Exception as e:
            logger.error(f"ClickHouse connection error: {str(e)}", exc_info=True)
            print(f"ClickHouse connection error: {e}")
            raise


    def insert(self, ticker: str, news_data: dict, cycle: int):
        """
        Insert aggregated news data for a single ticker using HTTP interface.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            news_data: Dictionary containing aggregated news data:
                - titles: List of news headlines
                - timestamps: List of publication timestamps
                - sentiment_scores: List of sentiment scores
                - relevance_scores: List of relevance scores
                - weighted_avg_sentiment: Weighted average sentiment score
                - news_url: URL to news article
            cycle: Processing cycle number for tracking
        """
        try:
            # Create JSON row in JSONEachRow format (ClickHouse requirement)
            json_row = json.dumps({
                "symbol": ticker,
                "news_titles": news_data["titles"],
                "news_timestamps": news_data["timestamps"],
                "sentiment_scores": news_data["sentiment_scores"],
                "relevance_scores": news_data["relevance_scores"],
                "weighted_avg_sentiment": news_data["weighted_avg_sentiment"],
                "news_url": news_data["news_url"],
                "cycle": cycle
            })

            # Insert using ClickHouse HTTP interface with JSONEachRow format
            query = "INSERT INTO market_data.sentiment_stream FORMAT JSONEachRow"
            response = requests.post(
                self.url,
                params={"query": query},
                data=json_row,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=10
            )

            if response.status_code == 200:
                headline_count = len(news_data['titles'])
                logger.info(
                    f"Inserted aggregated data for {ticker} → ClickHouse "
                    f"({headline_count} headlines, cycle {cycle})"
                )
                print(f"✓ Inserted aggregated data for {ticker} → ClickHouse ({headline_count} headlines)")

            else:
                logger.error(
                    f"HTTP insert failed for {ticker}: Status {response.status_code}, "
                    f"Response: {response.text[:200]}"
                )
                print(f"✗ HTTP insert failed for {ticker}: Status {response.status_code}, Response: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"ClickHouse insert request failed for {ticker}: {str(e)}", exc_info=True)
            print(f"✗ ClickHouse insert failed for {ticker}: {e}")
        except Exception as e:
            logger.error(f"ClickHouse insert failed for {ticker}: {str(e)}", exc_info=True)
            print(f"✗ ClickHouse insert failed for {ticker}: {e}")
            import traceback
            traceback.print_exc()


class NewsIngestionService:
    """
    Main service for ingesting financial news, performing sentiment analysis,
    and publishing results to Kafka and ClickHouse.
    
    This service:
    1. Fetches news from Alpha Vantage API for multiple topics
    2. Aggregates news by stock ticker
    3. Calculates weighted sentiment scores
    4. Publishes to Kafka and stores in ClickHouse
    5. Runs in cycles (every 2 hours) with dynamic date ranges
    """
    
    def __init__(self):
        """Initialize news ingestion service with Kafka producer and dependencies."""
        kafka_broker = os.getenv("KAFKA_BROKER")
        if not kafka_broker:
            logger.error("KAFKA_BROKER environment variable not set")
            raise ValueError("KAFKA_BROKER environment variable is required")
        
        logger.info(f"Initializing Kafka producer (broker: {kafka_broker})")
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_broker,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,  # Retry failed sends up to 3 times
            linger_ms=100  # Wait 100ms to batch messages
        )
        self.hash_set = set()  # Track processed news items to avoid duplicates
        self.ch = ClickHouseNewsWriter()
        self.timestamp_manager = TimestampManager()
        logger.info("NewsIngestionService initialized")

    def _hash(self, title: str, url: str) -> str:
        return hashlib.sha256(f"{title}{url}".encode()).hexdigest()

    def fetch_by_topic(self, topic: str, start_date: str, end_date: str):
        """
        Fetch news articles from Alpha Vantage API for a specific topic.
        
        Args:
            topic: News topic (e.g., 'technology', 'finance', 'blockchain')
            start_date: Start date in YYYYMMDDTHHMM format
            end_date: End date in YYYYMMDDTHHMM format
        
        Returns:
            List of news feed items, or empty list on error/rate limit
        """
        logger.info(f"Fetching news for topic '{topic}' (from {start_date} to {end_date})")
        print(f"Fetching → {topic} (from {start_date} to {end_date})")

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": topic,
            "limit": 200,  # Maximum articles per request
            "time_from": start_date,
            "time_to": end_date,
            "apikey": ALPHA_API_KEY
        }
        
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if r.status_code != 200:
                logger.warning(f"API returned status {r.status_code}: {r.text[:200]}")
                print(f"HTTP {r.status_code}: {r.text[:200]}")
                return []
            
            data = r.json()
            logger.debug(f"API Response keys: {list(data.keys())}")

            # Handle API rate limiting
            if "Note" in data:
                logger.warning(f"Rate limit message: {data['Note']}")
                logger.info("Rate limit detected → sleeping 65 seconds")
                print(f"Rate limit message: {data['Note']}")
                print("Rate limit → sleep 65s")
                time.sleep(65)
                return []

            if "Information" in data:
                logger.warning(f"API Information: {data['Information']}")
                logger.info("API issue detected → sleeping 65 seconds")
                print(f"API Information: {data['Information']}")
                print("Rate limit or API issue → sleep 65s")
                time.sleep(65)
                return []

            if "Error Message" in data:
                logger.error(f"API Error: {data['Error Message']}")
                print(f"API Error: {data['Error Message']}")
                return []

            feed = data.get("feed", [])
            if not feed:
                logger.warning("No feed data in API response")
                logger.debug(f"Full response: {json.dumps(data, indent=2)[:500]}")
                print("Warning: No feed data in response")
                print(f"Full response: {json.dumps(data, indent=2)[:500]}")

            logger.info(f"Fetched {len(feed)} news items for topic '{topic}'")
            return feed
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout while fetching topic '{topic}'")
            print(f"Request timeout for {topic}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for topic '{topic}': {str(e)}", exc_info=True)
            print(f"Request error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching topic '{topic}': {str(e)}", exc_info=True)
            print(f"Request error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def process_batch(self, feed: list):
        """
        Process a batch of news items and aggregate by stock ticker.
        
        This method:
        1. Filters news items to only include popular tickers
        2. Deduplicates news items using hash-based tracking
        3. Aggregates sentiment and relevance scores by ticker
        4. Calculates weighted average sentiment scores
        
        Args:
            feed: List of news items from Alpha Vantage API
        
        Returns:
            Dictionary mapping ticker symbols to aggregated news data:
            {
                "AAPL": {
                    "titles": [...],
                    "timestamps": [...],
                    "sentiment_scores": [...],
                    "relevance_scores": [...],
                    "weighted_avg_sentiment": 0.75,
                    "news_url": "..."
                },
                ...
            }
        """
        ticker_news = {}
        logger.debug(f"Processing batch of {len(feed)} news items")

        for item in feed:
            title = item.get("title")
            url = item.get("url")
            published = item.get("time_published")

            if not all([title, url, published]):
                continue

            # Check for duplicates
            key = self._hash(title, url)
            if key in self.hash_set:
                continue
            self.hash_set.add(key)

            # Process each ticker mentioned in this news item
            ticker_sentiments = item.get("ticker_sentiment", [])
            if not ticker_sentiments:
                continue

            for ticker_info in ticker_sentiments:
                ticker = ticker_info.get("ticker")
                if not ticker or ticker not in top_240_popular_tickers:
                    continue

                relevance = float(ticker_info.get("relevance_score", 0.0))
                sentiment = float(ticker_info.get("ticker_sentiment_score", 0.0))

                # Initialize ticker entry if doesn't exist
                if ticker not in ticker_news:
                    ticker_news[ticker] = {
                        "titles": [],
                        "timestamps": [],
                        "sentiment_scores": [],
                        "relevance_scores": [],
                        "news_url": url  # Store the most recent URL
                    }

                # Append news data
                ticker_news[ticker]["titles"].append(title)
                ticker_news[ticker]["timestamps"].append(published)
                ticker_news[ticker]["sentiment_scores"].append(sentiment)
                ticker_news[ticker]["relevance_scores"].append(relevance)

        # Calculate weighted average sentiment for each ticker
        for ticker, data in ticker_news.items():
            sentiments = data["sentiment_scores"]
            relevances = data["relevance_scores"]

            # Weighted average: sum(sentiment * relevance) / sum(relevance)
            total_weighted = sum(s * r for s, r in zip(sentiments, relevances))
            total_weight = sum(relevances)

            data["weighted_avg_sentiment"] = total_weighted / total_weight if total_weight > 0 else 0.0

        return ticker_news

    def run(self):
        """
        Main execution loop for news ingestion service.
        
        This method:
        1. Starts timestamp consumer for dynamic date ranges
        2. Waits for initial timestamp (with timeout)
        3. Runs continuous ingestion cycles (every 2 hours)
        4. Fetches news for all topics
        5. Aggregates and publishes results
        """
        logger.info("Starting News Ingestion Service with Dynamic Date Range")
        print("Starting Ultra-Efficient News Ingestion with Dynamic Date Range")

        # Start timestamp consumer in background thread
        logger.info("Starting timestamp consumer")
        self.timestamp_manager.start_consuming()

        # Wait for first timestamp to arrive (with timeout)
        # This ensures we have a valid date range before starting ingestion
        logger.info("Waiting for timestamp data from Kafka...")
        print("Waiting for timestamp data from Kafka...")


        while True:
            start_date, end_date = self.timestamp_manager.get_date_range()
            if start_date and end_date:
                logger.info(f"Initial date range set: {start_date} to {end_date}")
                print(f"Initial date range set: {start_date} to {end_date}")
                break
            time.sleep(2)


        cycle = 0
        try:
            logger.info("Starting main ingestion loop")
            while True:
                cycle += 1

                # Get current date range (may have been updated by timestamp consumer)
                start_date, end_date = self.timestamp_manager.get_date_range()
                if not start_date or not end_date:
                    # Fallback to default if timestamp manager hasn't received data
                    logger.warning("No date range available, waiting for timestamp")
                    while True:
                        start_date, end_date = self.timestamp_manager.get_date_range()
                        if start_date and end_date:
                            break
                        time.sleep(2)

                # Log cycle start
                cycle_start_time = datetime.now()
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Cycle {cycle} started at {cycle_start_time:%Y-%m-%d %H:%M:%S}")
                logger.info(f"Using date range: {start_date} to {end_date}")
                logger.info(f"{'=' * 60}")
                print(f"\n{'=' * 60}")
                print(f"Cycle {cycle} | {cycle_start_time:%Y-%m-%d %H:%M:%S}")
                print(f"Using date range: {start_date} to {end_date}")
                print(f"{'=' * 60}")

                all_ticker_news = {}

                for topic in TOPICS:
                    logging.info(f"\n--- Fetching topic: {topic} ---")
                    print(f"\n--- Fetching topic: {topic} ---")
                    feed = self.fetch_by_topic(topic, start_date, end_date)
                    logging.info(f"Received {len(feed)} news items for {topic}")
                    print(f"Received {len(feed)} news items for {topic}")

                    batch_ticker_news = self.process_batch(feed)
                    print(f"Processed {len(batch_ticker_news)} unique tickers from {topic}")
                    logging.info(f"Processed {len(batch_ticker_news)} unique tickers from {topic}")

                    # Merge with existing ticker news
                    for ticker, news_data in batch_ticker_news.items():
                        if ticker not in all_ticker_news:
                            all_ticker_news[ticker] = news_data
                        else:
                            # Append to existing ticker data
                            all_ticker_news[ticker]["titles"].extend(news_data["titles"])
                            all_ticker_news[ticker]["timestamps"].extend(news_data["timestamps"])
                            all_ticker_news[ticker]["sentiment_scores"].extend(news_data["sentiment_scores"])
                            all_ticker_news[ticker]["relevance_scores"].extend(news_data["relevance_scores"])

                            # Recalculate weighted average
                            sentiments = all_ticker_news[ticker]["sentiment_scores"]
                            relevances = all_ticker_news[ticker]["relevance_scores"]
                            total_weighted = sum(s * r for s, r in zip(sentiments, relevances))
                            total_weight = sum(relevances)
                            all_ticker_news[ticker][
                                "weighted_avg_sentiment"] = total_weighted / total_weight if total_weight > 0 else 0.0

                    # time.sleep(15)  # Safe for free tier

                print(f"\n--- Publishing Results ---")
                print(f"Total unique tickers to publish: {len(all_ticker_news)}")
                logging.info(f"\n--- Publishing Results ---")
                logging.info(f"Total unique tickers to publish: {len(all_ticker_news)}")

                # Send aggregated data to Kafka and ClickHouse
                published_count = 0
                for ticker, news_data in all_ticker_news.items():
                    message = {
                        "ticker": ticker,
                        "news_titles": news_data["titles"],
                        "news_timestamps": news_data["timestamps"],
                        "sentiment_scores": news_data["sentiment_scores"],
                        "relevance_scores": news_data["relevance_scores"],
                        "weighted_avg_sentiment": news_data["weighted_avg_sentiment"],
                        "news_url": news_data["news_url"],
                        "cycle": cycle
                    }

                    self.producer.send(KAFKA_TOPIC, value=message)
                    self.ch.insert(ticker, news_data, cycle)
                    published_count += 1

                    if published_count % 10 == 0:
                        print(f"Published {published_count}/{len(all_ticker_news)} tickers...")
                        logging.info(f"Published {published_count}/{len(all_ticker_news)} tickers...")
                        

                print(f"\n{'=' * 60}")
                print(f"Cycle {cycle} complete - Published {published_count} tickers")
                print(f"Next cycle in 2 hours...")
                print(f"{'=' * 60}\n")
                
                logging.info(f"\n{'=' * 60}")
                logging.info(f"Cycle {cycle} complete - Published {published_count} tickers")
                logging.info(f"Next cycle in 2 hours...")
                logging.info(f"{'=' * 60}\n")
                time.sleep(7200)

        except KeyboardInterrupt:
            logger.info("Service interrupted by user (KeyboardInterrupt)")
            print("\nGraceful shutdown...")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {str(e)}", exc_info=True)
            raise
        finally:
            # Cleanup resources
            logger.info("Cleaning up resources...")
            self.timestamp_manager.stop()
            self.producer.flush()
            self.producer.close()
            logger.info("Producer closed. Service shutdown complete.")
            print("Producer closed. Bye!")

if __name__ == "__main__":
    service = NewsIngestionService()
    service.run()