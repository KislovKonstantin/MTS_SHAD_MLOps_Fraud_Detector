import logging
import os
import pandas as pd
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

def load_model(model_path):
    logger.info(f'Loading model from {model_path}')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def make_pred(processed_df, model, source_info="kafka"):
    y_proba = model.predict_proba(processed_df)[:, 1]
    submission = pd.DataFrame({
        'score': y_proba,
        'fraud_flag': (y_proba > 0.5) * 1,
    })
    logger.info(f'Prediction done for data from {source_info}')
    return submission, y_proba