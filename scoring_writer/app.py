import json
import logging
import os
import psycopg2
from confluent_kafka import Consumer
from prometheus_client import start_http_server, Summary, Counter, Histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_WRITE_TIME = Summary('scoring_db_write_seconds', 'Время записи в БД')
SCORING_COUNT = Counter('scorings_total', 'Количество записанных скорингов')
FRAUD_SCORE_HISTOGRAM = Histogram('scoring_fraud_score', 'Распределение скоров',
                                  buckets=[i/50.0 for i in range(51)])
FRAUD_COUNT = Counter('fraud_detected_total', 'Количество мошеннических транзакций')

def get_db_config():
    return {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id SERIAL PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                score FLOAT NOT NULL,
                fraud_flag INT NOT NULL,
                us_state TEXT,
                merch TEXT,
                cat_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        logger.info("Table 'scores' ready.")

def clean_str(s):
    if s is None:
        return None
    s = s.replace("'", "`")
    return s

@DB_WRITE_TIME.time()
def insert_score(conn, data):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO scores (transaction_id, score, fraud_flag, us_state, merch, cat_id)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (data["transaction_id"], data["score"], data["fraud_flag"],
              data.get("us_state"), clean_str(data.get("merch")), data.get("cat_id")))
        conn.commit()
    SCORING_COUNT.inc()
    FRAUD_SCORE_HISTOGRAM.observe(data["score"])
    if data["fraud_flag"] == 1:
        FRAUD_COUNT.inc()

def run_consumer():
    start_http_server(8001)
    logger.info("Prometheus metrics on port 8001")

    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    scoring_topic = os.getenv("KAFKA_SCORING_TOPIC")
    consumer_config = {
        'bootstrap.servers': kafka_bootstrap,
        'group.id': 'scoring-writer',
        'auto.offset.reset': 'earliest',
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe([scoring_topic])

    db_config = get_db_config()
    conn = psycopg2.connect(**db_config)
    create_table(conn)

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            try:
                value = json.loads(msg.value().decode('utf-8'))
                insert_score(conn, value)
                logger.info(f"Written: {value['transaction_id']}")
            except Exception as e:
                logger.exception(f"Error: {e}")
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        consumer.close()
        conn.close()

if __name__ == "__main__":
    logger.info("Starting Scoring Writer...")
    run_consumer()