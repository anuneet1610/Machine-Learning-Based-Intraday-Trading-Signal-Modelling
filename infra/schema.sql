CREATE DATABASE IF NOT EXISTS market_data;
USE market_data;   

CREATE TABLE final_table
(
    symbol                            String,
    datetime                          DateTime,
    ts_ms                             Int64,
    close                             Float64,
    sigma_forecast                    Float64,
    arma_forecast                     Float64,
    ema_trend_filter_trend_up         UInt8,
    ema_trend_filter_trend_down       UInt8,
    long_term_bias_trend_up           UInt8,
    long_term_bias_trend_down         UInt8,
    macd_signal                       Float64,
    risk_adj_ret                      Float64,
    long_signal                       UInt8,
    short_signal                      UInt8,
    rsi_timing                        Float64,
    weighted_avg_sentiment_api        Float64,
    weighted_avg_sentiment_distilbert Float64,
    hour                              UInt8,
    day_of_week                       UInt8,
    day_of_month                      UInt8,
    month                             UInt8,
    quarter                           UInt8,
    year                              UInt16,
    hour_sin                          Float64,
    hour_cos                          Float64,
    day_sin                           Float64,
    day_cos                           Float64,
    month_sin                         Float64,
    month_cos                         Float64,
    ret_1                             Float64,
    ret_2                             Float64,
    ret_3                             Float64,
    ret_6                             Float64,
    vol_10                            Float64,
    prob_big                          Float64,
    prob_up                           Float64,
    signal                            UInt8,
    direction                         Int8,
    trade_signal                      Int8
)
ENGINE = MergeTree()
ORDER BY (symbol, datetime);

CREATE TABLE sentiment_stream
(
    symbol                            String,
    news_titles                       Array(String),
    news_timestamps                   Array(String),
    api_sentiment_scores              Array(Float32),
    findistilbert_sentiment_scores    Array(Float32),
    relevance_scores                  Array(Float32),
    weighted_avg_sentiment_api        Float32,
    weighted_avg_sentiment_distilbert Float32,
    news_url                          String,
    news_start_time                   String,
    news_end_time                     String,
    cycle                             UInt32,
    inserted_at                       DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (symbol, cycle);
