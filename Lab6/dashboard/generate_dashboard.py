import sys
import os

sys.path.append('.')

from sklearn.ensemble import GradientBoostingRegressor
from explainerdashboard import *
import pandas as pd
import pickle
import logging

from model import get_model, split_data
from eda import get_encoded_data

logger = logging.getLogger(__name__)

def main():
    try:
        model = get_model()
        logger.info("Модель загружена успешно")
        
        df = get_encoded_data()
        X_train, X_test, y_train, y_test = split_data(df)
        
        X_test_sample = X_test[:100]
        y_test_sample = y_test[:100]
        
        explainer = RegressionExplainer(
            model, 
            X_test_sample, 
            y_test_sample
        )
        
        logger.info("Explainer создан успешно")
        
        db = ExplainerDashboard(explainer)
        
        db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
        logger.info("Дашборд сохранен в dashboard.yaml")
        
    except Exception as e:
        logger.error(f"Ошибка при создании дашборда: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()