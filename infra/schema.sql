/* ============================================================
   DATABASE
   ============================================================ */
CREATE DATABASE IF NOT EXISTS market_data;
USE market_data;


/* ============================================================
   1. KAFKA INGESTION TABLE  
   (Raw stream from Kafka topic stock_table_ma)
   ============================================================ */
CREATE TABLE IF NOT EXISTS kafka_input (
    symbol String,
    timestamp DateTime,
    ts_ms UInt64,
    close Float32,
    sigma_forecast Float32,
    arma_forecast Float32,
    ema_trend_filter_trend_up UInt8,
    ema_trend_filter_trend_down UInt8,
    long_term_bias_trend_up UInt8,
    long_term_bias_trend_down UInt8,
    macd_signal Int32,
    risk_adj_ret Float32,
    long_signal UInt8,
    short_signal UInt8,
    rsi_timing Int32,
    pct_change Float32
)
ENGINE = Kafka
SETTINGS kafka_broker_list = 'localhost:9092',        -- Use docker service name in production
         kafka_topic_list = 'stock_table_trial',
         kafka_group_name = 'clickhouse_group',
         kafka_format = 'JSONEachRow',
         kafka_num_consumers = 1,                
         kafka_handle_error_mode = 'stream';       


/* ============================================================
   2. FINAL STORAGE TABLE  
   (Optimized, partitioned MergeTree table)
   ============================================================ */
CREATE TABLE IF NOT EXISTS final_table (
    symbol LowCardinality(String),
    timestamp DateTime,
    ts_ms UInt64,
    close Float32,
    sigma_forecast Float32,
    arma_forecast Float32,
    ema_trend_filter_trend_up UInt8,
    ema_trend_filter_trend_down UInt8,
    long_term_bias_trend_up UInt8,
    long_term_bias_trend_down UInt8,
    macd_signal Int32,
    risk_adj_ret Float32,
    long_signal UInt8,
    short_signal UInt8,
    rsi_timing Int32,
    pct_change Float32
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)          -- Monthly partitions → faster queries & cleanup
ORDER BY (symbol, timestamp)              -- Optimal for symbol-time series
TTL timestamp + INTERVAL 365 DAY          -- Auto-retention: 1 year (adjust as needed)
SETTINGS index_granularity = 8192;        -- Good balance for analytics workloads


/* ============================================================
   3. MATERIALIZED VIEW  
   (Pipes Kafka → final_table)
   ============================================================ */

-- Drop & recreate to avoid “already exists” errors
DROP VIEW IF EXISTS mv_kafka_to_final;

CREATE MATERIALIZED VIEW mv_kafka_to_final
TO final_table
AS
SELECT
    symbol,
    timestamp,
    ts_ms,
    close,
    sigma_forecast,
    arma_forecast,
    ema_trend_filter_trend_up,
    ema_trend_filter_trend_down,
    long_term_bias_trend_up,
    long_term_bias_trend_down,
    macd_signal,
    risk_adj_ret,
    long_signal,
    short_signal,
    rsi_timing,
    pct_change
FROM kafka_input;


/* ============================================================
   4. SENTIMENT STREAM TABLE  
   (Also optimized for time-series analytics)
   ============================================================ */
CREATE TABLE IF NOT EXISTS sentiment_stream (     symbol String,     news_titles Array(String),     news_timestamps Array(String),     sentiment_scores Array(Float64),     relevance_scores Array(Float64),     weighted_avg_sentiment Float64,     news_url String, cycle Float64 ) ENGINE = MergeTree() ORDER BY cycle;
