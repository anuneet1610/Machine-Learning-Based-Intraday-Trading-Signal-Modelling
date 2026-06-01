# Real-Time Intraday Trading Signal Platform

A real-time machine learning system that combines live stock market data and financial news sentiment to generate intraday trading signals.

Built using **Python**, **Pathway**, **Kafka**, **ClickHouse**, and **XGBoost**.

---

## Overview

Financial markets generate massive amounts of information every second. Traders often struggle to continuously monitor:

* Live stock price movements
* Technical indicators
* Breaking financial news
* Market sentiment changes

By the time information is manually gathered, analyzed, and acted upon, potential trading opportunities may already be gone.

This project addresses that problem by building an automated real-time intelligence pipeline that continuously ingests market data and news data, computes features, performs machine learning inference, and generates actionable trading alerts.

---

## Problem Statement

Traders and investors rely on information from multiple sources to make decisions.

These sources include:

* Historical price movements
* Technical indicators
* News articles
* Market sentiment

Manually monitoring and combining these data streams in real time is difficult and inefficient.

The objective of this project is to create a system capable of:

1. Continuously ingesting real-time market and news data.
2. Computing technical and sentiment-based features.
3. Making machine learning predictions on incoming data.
4. Generating alerts whenever significant price movement is predicted.

---

## System Architecture

```text
                   ┌──────────────────┐
                   │  Stock Service   │
                   └────────┬─────────┘
                            │
                            ▼
                     Kafka Topics
                            │
                            ▼
                ┌─────────────────────┐
                │ Calculation Service │
                │   (Pathway Stream)  │
                └────────┬────────────┘
                         │
                         ▼
                Technical Indicators
                         │
                         ▼
                ┌─────────────────────┐
                │    News Service     │
                └────────┬────────────┘
                         │
                         ▼
                  Sentiment Features
                         │
                         ▼
                ┌─────────────────────┐
                │  Decision Service   │
                │  XGBoost Inference  │
                └────────┬────────────┘
                         │
                         ▼
                    Predictions
                         │
                         ▼
                ┌─────────────────────┐
                │  Backend Service    │
                │ Firebase Alerts     │
                └────────┬────────────┘
                         │
                         ▼
                     End User
```

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

* Firebase Cloud Messaging (FCM)
* Logging Infrastructure

---

## Components

### Stock Service

Responsible for market data ingestion.

**Functions:**

* Streams stock market data
* Publishes events to Kafka
* Acts as the primary data source for downstream services

The data streamed can be accessed through this link: https://www.kaggle.com/datasets/anuneetgupta1610/high-frequency-stock-market-data

---

### Calculation Service

Built using Pathway's streaming computation engine.

**Functions:**

* Consumes market data streams
* Computes technical indicators
* Performs rolling aggregations
* Generates feature streams

**Example Features:**

* Returns
* Moving averages
* Volatility metrics
* Momentum indicators
* Price-based statistical features

---

### News Service

Processes financial news streams.

**Functions:**

* Consumes incoming news events
* Performs sentiment analysis
* Generates sentiment scores
* Publishes sentiment features
* Merges the sentiment features with feature data, applying decaying logic

---

### Decision Service

Hosts the machine learning inference engine.

**Functions:**

* Consumes technical features and sentiment features
* Runs trained XGBoost models
* Produces trading predictions

The service consists of 2 models:

* Big Move Predictor: Predicts the probability of a change of more than 20% in price
* Up/Down Predictor: Predicts the probability of the stock price going up

to estimate future intraday price movement.

---

### Backend Service

Responsible for notification delivery.

**Functions:**

* Consumes prediction outputs
* Detects significant signals
* Sends alerts through Firebase Cloud Messaging

Alerts are generated whenever the model predicts a price movement exceeding configured thresholds.

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

* Sentiment scores
* News impact indicators
* Aggregated sentiment signals
* Decaying sentiment across time

---

### Model

The prediction engine uses:

* **XGBoost**

The model receives both technical and sentiment features and predicts future intraday price movement.

---

## Real-Time Data Flow

1. Market data enters through the Stock Service.
2. Kafka distributes events across services.
3. Calculation Service computes technical indicators.
4. News Service generates sentiment features.
5. Decision Service performs model inference.
6. Predictions are stored and published.
7. Backend Service sends trading alerts.
8. Data is persisted in ClickHouse for analysis.

---

## Model Evaluation

Several XGBoost models were trained and evaluated using historical market and sentiment data. While the predictive performance remains modest (which is expected for short-horizon market forecasting), the project successfully demonstrates:

- Real-time feature generation
- Streaming inference pipelines
- Integration of technical and sentiment signals
- Automated alert generation

---

## Repository Structure

```text
.
├── backend/
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
kafka_input
mv_kafka_to_final
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

### Backend Service

```bash
python3 main.py
```

Logs will be written to the logs directory.

---

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
