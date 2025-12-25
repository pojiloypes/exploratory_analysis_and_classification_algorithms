import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import logging
import pickle
from eda import get_encoded_data

logger = logging.getLogger(__name__)

MODELS_PATH='models/'
MODEL_FILE_NAME='gb_model.cbm'

def get_model():
    try:
        if not os.path.exists(f'{MODELS_PATH}{MODEL_FILE_NAME}'):
            create_and_train_model()
        
        with open(f'{MODELS_PATH}{MODEL_FILE_NAME}', 'rb') as file:
            model = pickle.load(file)
        
        return model
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}", exc_info=True)
        raise


def create_and_train_model():
    try:
        df = get_encoded_data()
        x_train, x_test, y_train, y_test = split_data(df)
        
        gb = GradientBoostingRegressor(learning_rate=0.05, max_depth=4, n_estimators=100, random_state=42)
        gb.fit(x_train, y_train)
        
        test_score = gb.score(x_test, y_test)
        logger.info(f"Обучение модели завершено. R^2 Score: {test_score}")

        with open(f'{MODELS_PATH}{MODEL_FILE_NAME}', 'wb') as file:
            pickle.dump(gb, file)
        
        logger.info("Новая модель сохранена успешно.")

    except Exception as e:
        logger.error(f"Ошибка создания и обучения модели: {e}", exc_info=True)
        raise
    
def split_data(df: pd.DataFrame):
    try:
        x = df.drop(['Rings'], axis=1)
        y = df.Rings

        y_bins = pd.qcut(y, q=3, labels=False)
        y_bins.value_counts()
        
        return train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y_bins)
        
    except Exception as e:
        logger.error(f"Ошибка разделения данных: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    create_and_train_model()