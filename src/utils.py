import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def get_feature_importance(model, feature_names, top_n=5, json_path='app/output/feature_importance_top5.json'):
    logger.info('Starting to get feature importances...')
    importance = model.get_feature_importance()
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    fi_df = fi_df.sort_values('importance', ascending=False).head(top_n)
    top_features = dict(zip(fi_df['feature'], fi_df['importance']))

    with open(json_path, 'w') as f:
        json.dump(top_features, f, indent=4)

    logger.info('Feature importances saved in %s', json_path)


def get_prediction_density_plot(model, df, output_path='app/output/prediction_density.png', bins=50):
    logger.info('Getting probabilities for scores plot...')
    proba = model.predict_proba(df)[:, 1]

    plt.figure(figsize=(10, 6))
    sns.histplot(proba, bins=bins, kde=True, stat='density', color='steelblue', alpha=0.6)

    plt.xlabel('Predicted probability (fraud)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of predicted scores', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    logger.info('Scores distribution plot saved in %s', output_path)