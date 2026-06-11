# Machine Learning Based Intraday Trading Signal Modelling

A real-time machine learning system that combines live stock market data and financial news sentiment to generate intraday trading signals.

Built using **Python**, **Pathway**, **Kafka**, **ClickHouse**, and **XGBoost**.

---

## Overview

Financial markets generate massive amounts of information every second. 

By the time information is manually gathered, analyzed, and acted upon, potential trading opportunities may already be gone.

This project addresses that problem by building an automated real-time intelligence pipeline that continuously ingests market data and news data, computes features, performs machine learning inference, and generates actionable trading alerts.

---

## Problem Statement

Traders and investors rely on information from multiple sources to make decisions.

These sources include:

* Historical price movements
* Technical indicators
* News articles

Manually monitoring and combining these data streams in real time is difficult and inefficient.

The objective of this project is to create a system capable of:

1. Continuously ingesting real-time market and news data.
2. Computing technical and sentiment-based features.
3. Making machine learning predictions on incoming data.
4. Generating alerts whenever significant price movement is predicted.

---

## System Architecture

<img width="2983" height="1980" alt="Final_Architecture" src="https://github.com/user-attachments/assets/49402f5f-67c5-409c-86eb-68a624844b6d" />

---

## Technology Stack

### Core Technologies

* Python
* Pathway
* Apache Kafka
* ClickHouse
* XGBoost
* Docker

### Additional Services

* Logging Infrastructure

---

## Components

### Stock Service

Responsible for market data ingestion.

**Functions:**

* Replays the 5-min stock market data stored in a CSV file
* Publishes it to Kafka topic "stock_price_data"
* Publishes the timestamp of current stock ticker to Kafka topic "stock_timestamp"
* Acts as the primary data source for downstream services

The data streamed can be accessed through this link: https://www.kaggle.com/datasets/anuneetgupta1610/high-frequency-stock-market-data

---

### Calculation Service

Built using Pathway's streaming computation engine.

**Functions:**

* Consumes market data streams from Kafka topic "stock_price_data"
* Computes technical indicators
* Performs rolling aggregations
* Publishes the calculated metrics to Kafka topic "stock_metrics"

**Features:**

* Lagging Returns
* Moving averages
* Volatility metrics
* Momentum indicators
* Time-based statistical features

---

### News Service

Processes financial news streams.

**Functions:**

* Uses AlphaVantage API to fetch news as well as sentiment scores, based on stock symbol and time duration
* Consumes the timestamp from Kafka topic "stock_timestamp"
* Fetches news sentiment twice a day, once at the starting of trading day (14:30), and once at the end (20:55)
* For each fetch, a window of 10 days is considered, ending at current time
* If the current timestamp is neither 14:30, nor 20:55, then the latest news for that symbol is taken
* The news fetched is stored in ClickHouse table "sentiment_stream"
* Decay weighted-average sentiment is calculated per stock, using the latest news fetched. (Formula given below)
```text
Decay_i = e^(-hours_since_i / τ)     (τ = decay constant)

WeightedAvg = Σ(Sentiment_i × Relevance_i × Decay_i)
              --------------------------------------
                      Σ(Relevance_i × Decay_i)
```
* Consumes the stock market features from Kafka topic "stock_metrics"
* Merges the stock market features with the decay weighted-average sentiment values
* Publishes the final stock and news data to Kafka topic "metrics_data"

---

### Decision Service

Hosts the machine learning inference engine.

**Functions:**

* Consumes technical features and sentiment features from Kafka topic "metrics_data"
* Runs trained XGBoost models
* Produces trading predictions

The service consists of 2 models:

* Big Move Predictor: Predicts the probability of a change of more than 20% in price
* Up/Down Predictor: Predicts the probability of the stock price going up

to estimate future intraday price movement.

Alerts are generated when the Big Move and Up/Down predictions exceed the following threshold:
```text
* Probability of (|%age change| > 20%) > 0.35
   AND
* Probability of price going up > 53% OR < 42%
```

---

### ClickHouse

Acts as the analytical storage layer.

**Functions:**

* Stores processed market data
* Stores sentiment streams
* Stores prediction outputs
* Supports fast analytical queries

---

## Machine Learning Pipeline

### Input Features

#### Technical Features

Generated from historical market data:

* Lagged returns
* Rolling statistics
* Volatility measures
* Momentum indicators
* Trend-based features

#### Fundamental Features

Generated from financial news:

* Decay weighted-average sentiment scores

---

### Model

The prediction engine uses:

* 2 **XGBoost** models

The model receives both technical and sentiment features and predicts future intraday price movement.

---

## Real-Time Data Flow

1. Market data enters through the Stock Service.
2. Kafka distributes events across services.
3. Calculation Service computes technical indicators.
4. News Service generates sentiment features.
5. Decision Service performs model inference.
6. Predictions are stored and published.
7. Data is persisted in ClickHouse for analysis.

---

## Model Evaluation

Several XGBoost models were trained and evaluated using historical market and sentiment data. 

The up/down prediction model achieved:
- Accuracy: 51.43%
- F1 Score: 64.56%
- ROC AUC: 50.47%

The big move predictor model achieved: 
- Accuracy: 83.03%
- F1 Score: 38.82%
- ROC AUC: 82.68%

While the predictive performance remains modest (which is expected for short-horizon market forecasting), the project successfully demonstrates:

- Real-time feature generation
- Streaming inference pipelines
- Integration of technical and sentiment signals

---

## Performance Metrics

Average end-to-end latency: ~3.5 sec

Computing rolling windows for feature calculation acccounts for almost 95% of the end-to-end latency.

---

## Design Choices

Why Kafka?

- Decouples producers and consumers
- Fault tolerance
- Scalable event streaming

Why Pathway?

- Real-time streaming computations
- Native support for rolling aggregations
- Lower complexity than custom stream processing

Why ClickHouse?

- Fast analytical queries
- Efficient columnar storage

---

## Repository Structure

```text
.
├── calc_service/
├── decision_service/
├── infra/
├── news_service/
├── stock_service/
├── logs/
└── README.md
```

---

# Setup Guide

## Requirements

### 1. ClickHouse

Install ClickHouse:

https://clickhouse.com/docs/getting-started/quick-start/oss

### 2. Docker

Install Docker:

https://docs.docker.com/engine/install/

### 3. Python

Recommended versions:

* Python 3.12
* Python 3.13

Download:

https://www.python.org

---

## Step 1: Start ClickHouse Server

Navigate to your ClickHouse installation directory.

Start the server:

```bash
./clickhouse server
```

Open a second terminal and start the client:

```bash
./clickhouse client
```

---

## Step 2: Create Virtual Environments

Inside each service directory:

```bash
python3 -m venv env
source env/bin/activate
```

Windows PowerShell:

```powershell
env\Scripts\activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 3: Create Database Schema

Navigate to:

```text
infra/
```

Run:

```bash
python3 main.py
```

This creates the `market_data` database and required tables.

Verify:

```sql
SHOW DATABASES;
```

Expected output should include:

```text
market_data
```

Then run:

```sql
USE market_data;
SHOW TABLES;
```

Expected tables:

```text
final_table
sentiment_stream
```

---

## Step 4: Start Kafka

Navigate to:

```text
infra/
```

Run:

```bash
docker compose up
```

If necessary:

```bash
sudo docker compose up
```

---

## Step 5: Start Microservices

The services may be started in any order, but **Stock Service should be started last**.

Recommended order:

### Calculation Service

```bash
python3 main.py
```

This starts the Pathway streaming pipeline.

---

### Decision Service

```bash
python3 xgboost_mdl_inf.py
```

Expected output:

```text
Waiting for messages...
```

---

### News Service

```bash
python3 main.py
```

The service will wait for timestamps generated from upstream services.

---

### Stock Service

```bash
python3 main.py
```

Once market events begin flowing, all downstream services will start processing data.

---

## Future Improvements

* Automated model retraining
* Portfolio management layer
* Risk management engine
* Multi-asset support
* Real-time monitoring dashboards
* Advanced feature engineering
* Latency monitoring and observability
* Trade execution integration
