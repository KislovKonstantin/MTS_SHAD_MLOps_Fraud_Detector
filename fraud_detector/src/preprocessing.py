import pandas as pd
import numpy as np
import pickle
import logging
from geopy.distance import great_circle

logger = logging.getLogger(__name__)

def create_full_name(df):
    if 'full_name' not in df.columns:
        df['full_name'] = df['name_1'].fillna('') + ' ' + df['name_2'].fillna('')
        df['full_name'] = df['full_name'].str.strip()
    return df

def add_time_features(df):
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df['year'] = dt.year
    df['month'] = dt.month
    df['day_of_month'] = dt.day
    df['day_of_week'] = dt.dayofweek
    month = df['month']
    df['season'] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]),
         month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        [1, 2, 3, 4], default=0
    )
    df.drop(columns='transaction_time', inplace=True)
    return df

def add_distance_features(df):
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']),
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    return df

def apply_target_encoding(df, encoder_pkl_path):
    with open(encoder_pkl_path, 'rb') as f:
        encoder = pickle.load(f)
    cat_cols = encoder.cols
    encoded = encoder.transform(df[cat_cols])
    for col in cat_cols:
        df[f'te_{col}'] = encoded[col]
    return df

def run_preproc(input_df, encoder_path='models/target_encoder.pkl', imputer_path='models/imputer.pkl'):
    df = input_df.copy()
    logger.info(f'Preprocessing start, shape {df.shape}')

    df = create_full_name(df)
    df = apply_target_encoding(df, encoder_path)
    df = add_time_features(df)
    df = add_distance_features(df)

    continuous_cols = ['amount', 'population_city', 'distance']
    with open(imputer_path, 'rb') as f:
        imputer = pickle.load(f)
    df_cont = imputer.transform(df[continuous_cols])
    df[continuous_cols] = df_cont

    for col in continuous_cols:
        df[col + '_log'] = np.log(df[col] + 1)

    final_columns = ['lat', 'lon', 'merchant_lat', 'merchant_lon', 'hour', 'year', 'month',
                     'day_of_month', 'day_of_week', 'season', 'te_merch', 'te_cat_id',
                     'te_full_name', 'te_gender', 'te_street', 'te_one_city', 'te_us_state',
                     'te_post_code', 'te_jobs', 'amount', 'population_city', 'distance',
                     'amount_log', 'population_city_log', 'distance_log']
    missing = set(final_columns) - set(df.columns)
    for col in missing:
        df[col] = 0
    df = df[final_columns]
    logger.info(f'Preprocessing done, shape {df.shape}')
    return df