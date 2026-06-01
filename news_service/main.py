"""
Financial News & Stock Sentiment Pipeline

This module combines two microservices into a single unified pipeline:

1. NewsIngestionService (news_service)
   - Fetches financial news from Alpha Vantage API for target tickers
     (AAPL, TSLA, NVDA, JPM, JNJ)
   - Runs FinDistilBERT sentiment analysis on headlines
   - Publishes results to Kafka topic 'News' and stores in ClickHouse
     (market_data.sentiment_stream)
   - Triggered by 14:30 / 20:55 timestamps from Kafka topic 'stock_timestamp'
   - After each ticker is processed, writes results into the shared SentimentStore

2. StockSentimentMerger (stock_sentiment_merger)
   - Consumes stock technical indicator records from Kafka topic
     'stock_calculation_table'
   - Enriches each record with the latest sentiment from the shared
     SentimentStore (written directly by NewsIngestionService — no ClickHouse
     lookup needed)
   - Publishes merged records to Kafka topic 'metrics_data'

Both services run concurrently via threads and share sentiment data through
SentimentStore, a thread-safe in-memory dict keyed by ticker symbol.

Usage:
    python main_merged.py                  # run both services (default)
    python main_merged.py --news-only      # run NewsIngestionService only
    python main_merged.py --merger-only    # run StockSentimentMerger only
"""

import argparse
import hashlib
import json
import logging
import math
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================================
# Environment & Shared Configuration
# ============================================================================
load_dotenv()

KAFKA_BROKER       = os.getenv("KAFKA_BROKER", "localhost:9092")
ALPHA_API_KEY      = os.getenv("API_KEY")
BASE_URL           = os.getenv("API_URL")

NEWS_KAFKA_TOPIC   = "News"
TIMESTAMP_TOPIC    = "stock_timestamp"
TARGET_TICKERS     = ["AAPL", "TSLA", "NVDA", "JPM", "JNJ"]

SENTIMENT_MODEL_NAME  = "KernAI/stock-news-distilbert"
SENTIMENT_BATCH_SIZE  = 32

MERGER_INPUT_TOPIC  = "stock_calculation_table"
MERGER_OUTPUT_TOPIC = "metrics_data"

# Exponential time-decay half-life for sentiment scores (in hours).
# A TAU of 12 means a news article from 12 hours ago has ~37% of its original weight.
SENTIMENT_DECAY_TAU = 12.0

# ============================================================================
# Logging
# ============================================================================
log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)


def _make_logger(name: str) -> logging.Logger:
    handler = logging.FileHandler(f"../logs/{name}.log", mode="a")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    return log


news_logger   = _make_logger("news_service")
merger_logger = _make_logger("stock_sentiment_merger")


# ============================================================================
# SentimentStore — shared in-memory bridge between the two services
# ============================================================================

class SentimentStore:
    """
    Thread-safe in-memory store that holds the latest processed sentiment
    for each ticker symbol.

    NewsIngestionService.run() calls update() after every ticker is processed.
    StockSentimentMerger.run() calls get() for every incoming stock record.

    Raw per-article arrays (scores, relevance, timestamps) are stored here so
    that StockSentimentMerger can apply exponential time-decay at merge time,
    using the actual stock record timestamp as the reference point for
    hours_since_news.  The pre-aggregated weighted averages computed at fetch
    time are also kept for archival / logging purposes but are NOT used by the
    merger — the merger always recomputes them with decay applied.

    Structure:
        {
          "AAPL": {
            "news_timestamp": "...",          # timestamp of most-recent article
            "headline": [...],
            "sentiment_score_api": 0.42,      # score of most-recent article
            "sentiment_score_llm": 0.38,
            "relevance_scores": 0.91,         # relevance of most-recent article
            # ── raw per-article arrays (used for decay computation) ──
            "all_timestamps":         [...],  # ISO strings, one per article
            "all_relevance_scores":   [...],  # floats, one per article
            "all_api_sentiments":     [...],  # floats, one per article
            "all_distilbert_sentiments": [...],
            # ── pre-aggregated (archival only, not used by merger) ──
            "weighted_avg_sentiment_api":        0.35,
            "weighted_avg_sentiment_distilbert": 0.29,
            "news_end_time": "20240115T2055",
          },
          ...
        }
    """

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def update(self, ticker: str, sentiment: dict) -> None:
        """Overwrite the latest sentiment entry for ticker (called by NewsIngestionService)."""
        with self._lock:
            self._store[ticker] = sentiment
        merger_logger.info(f"SentimentStore updated for {ticker}")

    def get(self, ticker: str) -> dict | None:
        """Return the latest sentiment dict for ticker, or None if not yet available."""
        with self._lock:
            return self._store.get(ticker)


# ============================================================================
# TimestampManager
# ============================================================================

class TimestampManager:
    """
    Polls the stock_timestamp Kafka topic and blocks until a 14:30 or 20:55
    timestamp arrives, then returns the 10-day lookback window for the API.
    """

    FETCH_TIMES = {(14, 30), (20, 55)}

    def __init__(self):
        if not KAFKA_BROKER:
            raise ValueError("KAFKA_BROKER environment variable is required")

        news_logger.info(f"Initializing TimestampManager consumer (broker: {KAFKA_BROKER})")
        self.consumer = KafkaConsumer(
            TIMESTAMP_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            group_id="news-timestamp-consumer",
            value_deserializer=lambda m: m.decode("utf-8"),
        )
        while not self.consumer.assignment():
            self.consumer.poll(timeout_ms=1000)
        self.consumer.seek_to_end(*self.consumer.assignment())
        news_logger.info("TimestampManager: seeked to end of topic")
        print("Seeked to end of stock_timestamp topic")

    def _parse_timestamp(self, s: str) -> datetime | None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(s)
        except Exception as e:
            news_logger.warning(f"Failed to parse timestamp '{s}': {e}")
            return None

    def _fmt(self, dt: datetime) -> str:
        return dt.strftime("%Y%m%dT%H%M")

    def wait_for_next_fetch(self) -> tuple[str, str]:
        while True:
            try:
                for _, records in self.consumer.poll(timeout_ms=1000).items():
                    for record in records:
                        dt = self._parse_timestamp(record.value)
                        if not dt or (dt.hour, dt.minute) not in self.FETCH_TIMES:
                            continue
                        end_date, start_date = self._fmt(dt), self._fmt(dt - timedelta(days=10))
                        news_logger.info(f"Fetch trigger {dt:%Y-%m-%d %H:%M:%S} — {start_date} to {end_date}")
                        return start_date, end_date
            except Exception as e:
                news_logger.error(f"TimestampManager poll error: {e}", exc_info=True)
                time.sleep(1)

    def stop(self):
        self.consumer.close()
        news_logger.info("TimestampManager consumer closed")


# ============================================================================
# ClickHouseNewsWriter  (archival write — merger no longer reads from here)
# ============================================================================

class ClickHouseNewsWriter:
    """Writes aggregated news sentiment to ClickHouse for archival purposes."""

    def __init__(self):
        self.url = os.getenv("CLICKHOUSE_URL")
        if not self.url:
            raise ValueError("CLICKHOUSE_URL environment variable is required")
        resp = requests.get(self.url, params={"query": "SELECT 1"}, timeout=5)
        if resp.status_code == 200:
            news_logger.info("ClickHouse HTTP connection established")
            print("ClickHouse HTTP connection established successfully")
        else:
            news_logger.warning(f"ClickHouse connection status: {resp.status_code}")

    def insert(self, ticker: str, news_data: dict, cycle: int):
        try:
            json_row = json.dumps({
                "symbol":                            ticker,
                "news_titles":                       news_data["titles"],
                "news_timestamps":                   news_data["timestamps"],
                "api_sentiment_scores":              news_data["api_sentiment_scores"],
                "findistilbert_sentiment_scores":    news_data["findistilbert_sentiment_scores"],
                "relevance_scores":                  news_data["relevance_scores"],
                "weighted_avg_sentiment_api":        news_data["weighted_avg_sentiment_api"],
                "weighted_avg_sentiment_distilbert": news_data["weighted_avg_sentiment_distilbert"],
                "news_url":                          news_data["news_url"],
                "news_start_time":                   news_data["news_start_time"],
                "news_end_time":                     news_data["news_end_time"],
                "cycle":                             cycle,
            })
            resp = requests.post(
                self.url,
                params={"query": "INSERT INTO market_data.sentiment_stream FORMAT JSONEachRow"},
                data=json_row,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=10,
            )
            if resp.status_code == 200:
                news_logger.info(f"Inserted {len(news_data['titles'])} headlines for {ticker} (cycle {cycle})")
                print(f"Inserted {ticker} -> ClickHouse ({len(news_data['titles'])} headlines)")
            else:
                news_logger.error(f"ClickHouse insert failed for {ticker}: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            news_logger.error(f"ClickHouse insert error for {ticker}: {e}", exc_info=True)


# ============================================================================
# NewsIngestionService
# ============================================================================

class NewsIngestionService:
    """
    Fetches news, scores sentiment, publishes to Kafka + ClickHouse, and
    writes the latest sentiment for each ticker into the shared SentimentStore
    so that StockSentimentMerger can access it directly in memory.
    """

    def __init__(self, sentiment_store: SentimentStore):
        self.sentiment_store = sentiment_store

        news_logger.info(f"Initializing Kafka producer (broker: {KAFKA_BROKER})")
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,
            linger_ms=100,
        )
        self.hash_set: set[str] = set()
        self.ch = ClickHouseNewsWriter()
        self.timestamp_manager = TimestampManager()

        news_logger.info(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}")
        print(f"Loading sentiment model: {SENTIMENT_MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        news_logger.info(f"Sentiment model loaded on {self.device}")
        print(f"Sentiment model ready on {self.device}")

    def _hash(self, title: str, url: str) -> str:
        return hashlib.sha256(f"{title}{url}".encode()).hexdigest()

    def _score_headlines_batch(self, titles: list[str]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(titles), SENTIMENT_BATCH_SIZE):
            chunk = titles[i: i + SENTIMENT_BATCH_SIZE]
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            scores.extend((probs[:, 2] - probs[:, 0]).tolist())
        return scores

    def fetch_by_ticker(self, ticker: str, start_date: str, end_date: str) -> list:
        news_logger.info(f"Fetching news for '{ticker}' ({start_date} to {end_date})")
        print(f"Fetching -> {ticker} ({start_date} to {end_date})")
        params = {
            "function": "NEWS_SENTIMENT", "tickers": ticker,
            "limit": 200, "time_from": start_date, "time_to": end_date, "apikey": ALPHA_API_KEY,
        }
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if r.status_code != 200:
                return []
            data = r.json()
            for key in ("Note", "Information"):
                if key in data:
                    news_logger.warning(f"API {key} — sleeping 65s")
                    time.sleep(65)
                    return []
            if "Error Message" in data:
                news_logger.error(f"API Error: {data['Error Message']}")
                return []
            feed = data.get("feed", [])
            news_logger.info(f"Fetched {len(feed)} items for '{ticker}'")
            return feed
        except requests.exceptions.Timeout:
            news_logger.error(f"Timeout fetching '{ticker}'")
            return []
        except Exception as e:
            news_logger.error(f"Error fetching '{ticker}': {e}", exc_info=True)
            return []

    def process_batch(self, ticker, feed, news_start_time=None, news_end_time=None):
        rows, news_url = [], None
        for item in feed:
            title, url, published = item.get("title"), item.get("url"), item.get("time_published")
            if not all([title, url, published]):
                continue
            key = self._hash(title, url)
            if key in self.hash_set:
                continue
            ticker_info = next((t for t in item.get("ticker_sentiment", []) if t.get("ticker") == ticker), None)
            if ticker_info is None:
                continue
            self.hash_set.add(key)
            if news_url is None:
                news_url = url
            rows.append((title, url, published, float(ticker_info.get("relevance_score", 0.0)), float(ticker_info.get("ticker_sentiment_score", 0.0))))

        if not rows:
            return None

        titles           = [r[0] for r in rows]
        sentiment_scores = self._score_headlines_batch(titles)
        relevance_scores = [r[3] for r in rows]
        api_sentiments   = [r[4] for r in rows]
        timestamps       = [r[2] for r in rows]
        total_weight     = sum(relevance_scores)

        return {
            "titles":                            titles,
            "timestamps":                        timestamps,
            "api_sentiment_scores":              api_sentiments,
            "findistilbert_sentiment_scores":    sentiment_scores,
            "relevance_scores":                  relevance_scores,
            "weighted_avg_sentiment_api":        sum(s*r for s,r in zip(api_sentiments, relevance_scores)) / total_weight if total_weight else 0.0,
            "weighted_avg_sentiment_distilbert": sum(s*r for s,r in zip(sentiment_scores, relevance_scores)) / total_weight if total_weight else 0.0,
            "news_url":        news_url,
            "news_start_time": news_start_time,
            "news_end_time":   news_end_time,
        }

    def run(self):
        news_logger.info("Starting NewsIngestionService — waiting for 14:30 / 20:55 timestamps")
        print("Starting NewsIngestionService — waiting for 14:30 / 20:55 timestamps")
        cycle = 0
        try:
            while True:
                print("Waiting for next 14:30 or 20:55 timestamp from Kafka...")
                start_date, end_date = self.timestamp_manager.wait_for_next_fetch()
                cycle += 1
                cycle_start = datetime.now()
                print(f"\n{'='*60}\nCycle {cycle} | {cycle_start:%Y-%m-%d %H:%M:%S} | {start_date} -> {end_date}\n{'='*60}")
                news_logger.info(f"Cycle {cycle} started — {start_date} to {end_date}")

                published_count = 0
                for ticker in TARGET_TICKERS:
                    print(f"\n--- Fetching ticker: {ticker} ---")
                    feed = self.fetch_by_ticker(ticker, start_date, end_date)
                    news_data = self.process_batch(ticker, feed, news_start_time=start_date, news_end_time=end_date)

                    if news_data is None:
                        news_logger.info(f"No valid news for {ticker}, skipping")
                        print(f"No valid news items for {ticker}, skipping.")
                        time.sleep(5)
                        continue

                    message = {
                        "ticker":                            ticker,
                        "news_titles":                       news_data["titles"],
                        "news_timestamps":                   news_data["timestamps"],
                        "api_sentiment_scores":              news_data["api_sentiment_scores"],
                        "findistilbert_sentiment_scores":    news_data["findistilbert_sentiment_scores"],
                        "relevance_scores":                  news_data["relevance_scores"],
                        "weighted_avg_sentiment_api":        news_data["weighted_avg_sentiment_api"],
                        "weighted_avg_sentiment_distilbert": news_data["weighted_avg_sentiment_distilbert"],
                        "news_url":                          news_data["news_url"],
                        "news_start_time":                   news_data["news_start_time"],
                        "news_end_time":                     news_data["news_end_time"],
                        "cycle":                             cycle,
                    }

                    # Publish to Kafka + persist to ClickHouse (archival)
                    self.producer.send(NEWS_KAFKA_TOPIC, value=message)
                    self.ch.insert(ticker, news_data, cycle)

                    # ── Write directly into shared store for StockSentimentMerger ──
                    # Raw per-article arrays are stored so the merger can apply
                    # exponential time-decay at merge time (using the stock
                    # record's timestamp as the reference point).
                    self.sentiment_store.update(ticker, {
                        "news_timestamp":                    news_data["timestamps"][-1] if news_data["timestamps"] else None,
                        "headline":                          news_data["titles"],
                        "sentiment_score_api":               news_data["api_sentiment_scores"][-1] if news_data["api_sentiment_scores"] else 0.0,
                        "sentiment_score_llm":               news_data["findistilbert_sentiment_scores"][-1] if news_data["findistilbert_sentiment_scores"] else 0.0,
                        "relevance_scores":                  news_data["relevance_scores"][-1] if news_data["relevance_scores"] else 0.0,
                        # raw arrays — used by merger for decay-weighted average
                        "all_timestamps":                    news_data["timestamps"],
                        "all_relevance_scores":              news_data["relevance_scores"],
                        "all_api_sentiments":                news_data["api_sentiment_scores"],
                        "all_distilbert_sentiments":         news_data["findistilbert_sentiment_scores"],
                        # pre-aggregated values kept for archival/logging only
                        "weighted_avg_sentiment_api":        news_data["weighted_avg_sentiment_api"],
                        "weighted_avg_sentiment_distilbert": news_data["weighted_avg_sentiment_distilbert"],
                        "news_end_time":                     news_data["news_end_time"],
                    })

                    news_logger.info(
                        f"Cycle {cycle} {ticker}: "
                        f"weighted_api={news_data['weighted_avg_sentiment_api']:.4f}, "
                        f"weighted_distilbert={news_data['weighted_avg_sentiment_distilbert']:.4f}"
                    )
                    published_count += 1
                    time.sleep(3)

                news_logger.info(f"Cycle {cycle} complete — {published_count} tickers published")
                print(f"\n{'='*60}\nCycle {cycle} complete — {published_count} tickers published\n{'='*60}\n")

        except KeyboardInterrupt:
            news_logger.info("NewsIngestionService interrupted by user")
            print("\nGraceful shutdown...")
        except Exception as e:
            news_logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            self.timestamp_manager.stop()
            self.producer.flush()
            self.producer.close()
            news_logger.info("NewsIngestionService shutdown complete")
            print("Producer closed. Bye!")


# ============================================================================
# StockSentimentMerger
# ============================================================================

def _normalise_timestamp(ts_value) -> str:
    if isinstance(ts_value, (int, float)):
        if ts_value > 1e10:
            ts_value /= 1000.0
        return datetime.fromtimestamp(ts_value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return str(ts_value).replace("T", " ").split(".")[0]


def _parse_article_timestamp(ts_str: str) -> datetime | None:
    """Parse an Alpha Vantage article timestamp (e.g. '20240115T143000') to a UTC datetime."""
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def _compute_decay_weighted_avg(
    raw_scores: list[float],
    relevance_scores: list[float],
    article_timestamps: list[str],
    reference_dt: datetime,
    tau: float,
) -> float:
    """
    Compute relevance-weighted average sentiment with exponential time-decay.

        effective_sentiment_i = raw_score_i * exp(-hours_since_news_i / tau)
        result = sum(effective_sentiment_i * relevance_i) / sum(relevance_i * decay_i)

    The denominator uses decay-adjusted relevance so the average stays in the
    same numeric range as the raw scores.  Falls back to 0.0 if all weights
    are zero or timestamps can't be parsed.
    """
    weighted_sum = 0.0
    weight_sum   = 0.0

    for score, relevance, ts_str in zip(raw_scores, relevance_scores, article_timestamps):
        article_dt = _parse_article_timestamp(ts_str)
        if article_dt is None:
            merger_logger.warning(f"Could not parse article timestamp '{ts_str}', skipping in decay calc")
            continue

        hours_since = max((reference_dt - article_dt).total_seconds() / 3600.0, 0.0)
        decay        = math.exp(-hours_since / tau)
        eff_weight   = relevance * decay

        weighted_sum += score * eff_weight
        weight_sum   += eff_weight

    return weighted_sum / weight_sum if weight_sum > 0.0 else 0.0


def _build_merged_record(data: dict, sentiment: dict | None, stock_dt: datetime) -> dict:
    merged = dict(data)
    if sentiment:
        # Recompute weighted averages with exponential time-decay applied to
        # each article's raw score, using the stock record timestamp as "now".
        decay_api = _compute_decay_weighted_avg(
            raw_scores          = sentiment["all_api_sentiments"],
            relevance_scores    = sentiment["all_relevance_scores"],
            article_timestamps  = sentiment["all_timestamps"],
            reference_dt        = stock_dt,
            tau                 = SENTIMENT_DECAY_TAU,
        )
        decay_distilbert = _compute_decay_weighted_avg(
            raw_scores          = sentiment["all_distilbert_sentiments"],
            relevance_scores    = sentiment["all_relevance_scores"],
            article_timestamps  = sentiment["all_timestamps"],
            reference_dt        = stock_dt,
            tau                 = SENTIMENT_DECAY_TAU,
        )

        merged["news_timestamp"]                    = sentiment["news_timestamp"]
        merged["headline"]                          = sentiment["headline"]
        merged["sentiment_score_api"]               = sentiment["sentiment_score_api"]
        merged["sentiment_score_llm"]               = sentiment["sentiment_score_llm"]
        merged["relevance_scores"]                  = sentiment["relevance_scores"]
        merged["weighted_avg_sentiment_api"]        = decay_api
        merged["weighted_avg_sentiment_distilbert"] = decay_distilbert
        merged["sentiment_found"]                   = True
    else:
        merged.update({
            "news_timestamp": None, "headline": None,
            "sentiment_score_api": None, "sentiment_score_llm": None,
            "relevance_scores": None, "weighted_avg_sentiment_api": None,
            "weighted_avg_sentiment_distilbert": None, "sentiment_found": False,
        })
    return merged


class StockSentimentMerger:
    """
    Consumes stock technical indicator records from Kafka, enriches each with
    the latest sentiment from the shared SentimentStore (no ClickHouse reads),
    and publishes merged records to Kafka topic 'metrics_data'.
    """

    def __init__(self, sentiment_store: SentimentStore):
        self.sentiment_store = sentiment_store

        merger_logger.info(f"Configuring Kafka consumer for '{MERGER_INPUT_TOPIC}'")
        self.consumer = KafkaConsumer(
            MERGER_INPUT_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            group_id="stock-sentiment-merger-group",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

        merger_logger.info(f"Configuring Kafka producer for '{MERGER_OUTPUT_TOPIC}'")
        self.producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        merger_logger.info("StockSentimentMerger initialized")
        print("StockSentimentMerger: waiting for messages...")

    def run(self):
        merger_logger.info("Starting StockSentimentMerger processing loop")
        try:
            for msg in self.consumer:
                data = msg.value
                print("Received:", data)

                data["symbol"] = data["symbol"].strip()
                symbol = data["symbol"]
                if not symbol:
                    merger_logger.warning("Empty symbol, skipping")
                    continue

                raw_ts = data.get("timestamp") or data.get("ts_ms")
                if raw_ts is None:
                    merger_logger.warning(f"No timestamp for {symbol}, skipping")
                    continue

                stock_ts = _normalise_timestamp(raw_ts)
                # Parse stock_ts into a timezone-aware datetime for decay calc
                try:
                    stock_dt = datetime.strptime(stock_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    stock_dt = datetime.now(tz=timezone.utc)
                    merger_logger.warning(f"Could not parse stock_ts '{stock_ts}' for decay; using now()")
                merger_logger.info(f"Processing {symbol} @ {stock_ts}")

                # ── Read directly from shared in-memory store ─────────────────
                sentiment = self.sentiment_store.get(symbol)

                if sentiment:
                    merger_logger.info(
                        f"[store] Sentiment for {symbol}: "
                        f"news_ts={sentiment['news_timestamp']}, "
                        f"score_api={sentiment['sentiment_score_api']:.4f}, "
                        f"score_llm={sentiment['sentiment_score_llm']:.4f} "
                        f"(decay-weighted averages will be computed at merge time)"
                    )
                    print(f"Sentiment found for {symbol}: {sentiment['news_timestamp']}")
                else:
                    merger_logger.warning(
                        f"[store] No sentiment yet for {symbol} — "
                        "NewsIngestionService may not have run this cycle"
                    )
                    print(f"No sentiment found for {symbol} @ {stock_ts}")

                merged = _build_merged_record(data, sentiment, stock_dt)
                if sentiment and merged["sentiment_found"]:
                    merger_logger.info(
                        f"[decay] {symbol}: "
                        f"decay_weighted_api={merged['weighted_avg_sentiment_api']:.4f}, "
                        f"decay_weighted_distilbert={merged['weighted_avg_sentiment_distilbert']:.4f} "
                        f"(TAU={SENTIMENT_DECAY_TAU}h)"
                    )
                try:
                    self.producer.send(MERGER_OUTPUT_TOPIC, value=merged)
                    self.producer.flush()
                    merger_logger.info(f"Sent merged record for {symbol} (sentiment_found={merged['sentiment_found']})")
                    print(f"Merged record sent for {symbol}")
                except Exception as e:
                    merger_logger.error(f"Failed to send for {symbol}: {e}", exc_info=True)

        except KeyboardInterrupt:
            merger_logger.info("StockSentimentMerger interrupted")
        except Exception as e:
            merger_logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            self.consumer.close()
            self.producer.close()
            merger_logger.info("StockSentimentMerger shutdown complete")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Financial News & Stock Sentiment Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--news-only",   action="store_true", help="Run only NewsIngestionService")
    group.add_argument("--merger-only", action="store_true", help="Run only StockSentimentMerger (store starts empty)")
    args = parser.parse_args()

    # Single shared store — the in-memory bridge between both services
    sentiment_store = SentimentStore()

    if args.news_only:
        NewsIngestionService(sentiment_store).run()
        return

    if args.merger_only:
        StockSentimentMerger(sentiment_store).run()
        return

    # Default: run both in parallel, sharing the same store
    news_service   = NewsIngestionService(sentiment_store)
    merger_service = StockSentimentMerger(sentiment_store)

    news_thread   = threading.Thread(target=news_service.run,   name="NewsIngestionService",   daemon=True)
    merger_thread = threading.Thread(target=merger_service.run, name="StockSentimentMerger",   daemon=True)

    print("Starting both services concurrently...")
    news_thread.start()
    merger_thread.start()

    try:
        while news_thread.is_alive() or merger_thread.is_alive():
            news_thread.join(timeout=1)
            merger_thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\nShutdown requested — threads will stop at next safe point.")


if __name__ == "__main__":
    main()