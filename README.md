# Machine Learning Based Intraday Trading Signal Modelling

# Problem
It is very difficult for traders to gather real time information from multiple sources, analyse them, and take appropriate action. This can lead to loss of potential profits.

# Approach
We propose a ML based real time system which can ingest information from news sources and stock market in real time, and make predictions using XGBoost model. It uses various technical indicators to interpret patterns in the past stock price data. The technical indicators as well as the fundamental indicator (news data) are fed to the XGBoost model, which makes predictions.

# Tech Stack
1. Python
2. Pathway (Python Library) (Used for building efficient data streaming systems)
3. Kafka
4. ClickHouse 

## User Guide

### Requirements:
1. Clickhouse Database: [https://clickhouse.com/docs/getting-started/quick-start/oss](https://clickhouse.com/docs/getting-started/quick-start/oss)
2. Docker: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
3. Python (3.12 or 3.13 preferred): [https://www.python.org](https://www.python.org)

### Steps to run the project

**1. Run the Clichouse Server:**  
Wherever the clickhouse database file installed, in a terminal `cd` into that folder. And run:
```bash
$ ./clickhouse server
```

In a new terminal window, go to the clickhouse directory and run the clickhouse client
```bash
$ ./clickhouse client
```

**2. Creating virtual envenvironments**  
In  all the directories present with this README file, there are requirements files and some python files, and also the files for environment variables. Open CLI in each folder and run the following commands to create virtual environments, and install the requirments.

```bash
$ python3 -m venv env
$ source env/bin/activate # for windows, env\Scripts\activate.ps1 (for powershell) or activate.bat (for command prompt)
$ pip3 install -r requirements.txt
```

**3. Creating the database and schema**  
Go to the `infra` directory, make sure that the virtual environment is activated and then run the following command:
```bash
$ python3 main.py
```
This will create the `market_data` clickhouse database and its required tables. You can verify this by going to the terminal window where clickhouse client is running and then run,
```bash
$ SHOW databases;
```

The output should include `market_data` database

and then run
```bash
$ USE market_data;
$ SHOW tables;
```

The output should be names of 4 tables:
1. final_table
2. kafka_input
3. mv_kafka_to_final
4. sentiment_stream

**4. Run the kafka server**
Go back to `infra` service and then run:
```bash
$ docker compose up # or sudo docker compose up if permission is denied
```
This will start the kafka server

**5. Running the microservices**
The microservices can be run in any order, except the fact that stock service must be run in the end.
We recommend to run the services in the alphabetical order itself.

So go to backend directory, make sure virtual env is activated and run
```bash
$ python3 main.py
```

This won't output anything yet, but you can see a logs folder where we will store all the logs

Then go to calc_service, with respective env activated and run
```bash
$ python3 main.py
```

This will take some time to run, and it will start the pathway computational stream but might show errors as we are not pushing anything to the required kafka topic yet. That will be done in the end in the stock service

Then go to decision service, with respective env activated and run
```bash
$ python3 xgboost_mdl_inf.py
```

This will output `Waiting for messages...`

Next go to news service, and do the same drill. Activate environment, and run:
```bash
$ python3 main.py
```
This will wait for timestamps, which are required from calculation service which in turn depends on stock service.

So finally, we go to the stock service, activate the environment and run
```bash
$ python3 main.py
```

This will take few seconds to run, but as soon as it starts pushing to kafka topic, outputs should start becoming visible in all the other microservices as well.

The main output that we won't to see is the alert messages being sent through Firebase Cloud Messagin service. This output might not be as frequent as outputs of other services, but it should send whenever the decision service finds a change of more than 2%, and pushes to the alert topic. The output of backend service would start with *'Alert sent'*.
