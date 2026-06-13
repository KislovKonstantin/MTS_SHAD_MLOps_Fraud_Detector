import json
import logging
import os
import sys

import pandas as pd
from confluent_kafka import Consumer, Producer
from prometheus_client import start_http_server, Summary, Counter, Histogram, Gauge

sys.path.append(os.path.abspath('./src'))
from preprocessing import run_preproc
from scorer import load_model, make_pred

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scoring")

PROCESSING_TIME = Summary('transaction_processing_seconds', 'Время обработки транзакции')
TRANSACTION_COUNT = Counter('transactions_total', 'Общее количество обработанных транзакций')
FRAUD_SCORE = Histogram('fraud_score', 'Распределение скоров мошенничества',
                        buckets=[i/50.0 for i in range(51)])
FRAUD_RATIO = Gauge('fraud_ratio', 'Соотношение мошеннических транзакций')

class FraudDetector:
    def __init__(self):
        self.consumer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'fraud-detector',
            'auto.offset.reset': 'earliest',
        }
        self.producer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer(self.producer_config)

        self.model = load_model('./models/catboost_model.cbm')
        self.total_transactions = 0
        self.fraud_transactions = 0

        start_http_server(8000)
        logger.info("Prometheus metrics on port 8000")

    @PROCESSING_TIME.time()
    def process_message(self, msg):
        try:
            data = json.loads(msg.value().decode('utf-8'))
            transaction_id = data['transaction_id']
            input_df = pd.DataFrame([data['data']])

            processed_df = run_preproc(
                input_df,
                encoder_path='./models/target_encoder.pkl',
                imputer_path='./models/imputer.pkl'
            )
            submission, y_proba = make_pred(processed_df, self.model, source_info="kafka")

            TRANSACTION_COUNT.inc()
            FRAUD_SCORE.observe(y_proba[0])
            self.total_transactions += 1
            if y_proba[0] > 0.5:
                self.fraud_transactions += 1
            if self.total_transactions > 0:
                FRAUD_RATIO.set(self.fraud_transactions / self.total_transactions)

            result = {
                'transaction_id': transaction_id,
                'score': float(y_proba[0]),
                'fraud_flag': int(y_proba[0] > 0.5),
                'us_state': data['data'].get('us_state', ''),
                'merch': data['data'].get('merch', ''),
                'cat_id': data['data'].get('cat_id', '')
            }
            self.producer.produce(SCORING_TOPIC, value=json.dumps(result).encode('utf-8'))
            self.producer.flush()
            logger.info(f"Processed {transaction_id}, score={y_proba[0]:.3f}")
            return True
        except Exception as e:
            logger.exception(f"Error: {e}")
            return False

    def run(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            self.process_message(msg)

if __name__ == "__main__":
    logger.info("Starting Fraud Detector...")
    detector = FraudDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        logger.info("Stopped")