import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

sys.path.append(os.path.abspath('./src'))

from preprocessing import run_preproc
from scorer import load_model, make_pred
from utils import get_feature_importance, get_prediction_density_plot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.model_path = './models/catboost_model.cbm'
        self.encoder_path = './models/target_encoder.pkl'
        self.imputer_path = './models/imputer.pkl'
        self.model = load_model(self.model_path)
        logger.info('Service initialized successfully')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path)
            logger.info('Raw data shape: %s', input_df.shape)

            processed_df = run_preproc(
                input_df,
                encoder_path=self.encoder_path,
                imputer_path=self.imputer_path
            )
            logger.info('Preprocessing completed. Shape: %s', processed_df.shape)

            submission = make_pred(processed_df, self.model, file_path)
            logger.info('Prediction completed')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            feature_importance_filename = f"feature_importance_top5_{timestamp}.json"
            prediction_density_plot_filename = f"prediction_density_{timestamp}_{os.path.splitext(os.path.basename(file_path))[0]}.png"

            feature_names = processed_df.columns.tolist()
            get_feature_importance(
                self.model,
                feature_names,
                json_path=os.path.join(self.output_dir, feature_importance_filename)
            )

            get_prediction_density_plot(
                self.model,
                processed_df,
                output_path=os.path.join(self.output_dir, prediction_density_plot_filename)
            )

            submission_path = os.path.join(self.output_dir, submission_filename)
            submission.to_csv(submission_path, index=False)
            logger.info('Submission saved to: %s', submission_path)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            raise

class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.csv'):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()

    observer = PollingObserver()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started on %s', service.input_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()