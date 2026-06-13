import pandas as pd
import logging
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

def load_model(model_path='models/catboost_model.cbm'):
    logger.info('Importing pretrained model...')
    model = CatBoostClassifier()
    model.load_model(model_path)

    logger.info('Pretrained model imported successfully...')
    return model

def make_pred(dt, model, path_to_file):
    model_th = 0.5
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (model.predict_proba(dt)[:, 1] > model_th) * 1
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    return submission