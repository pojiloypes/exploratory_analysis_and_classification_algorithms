import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle
import logging

logger = logging.getLogger(__name__)

RAW_DATA_PATH='data/abalone_raw.csv'
ENCODED_DATA_PATH='data/abalone_encoded.csv'
MODELS_PATH='models/'
ENCODER_FILE_NAME='encoder.pkl'

def get_encoded_data():
    try:
        if not os.path.exists(ENCODED_DATA_PATH):
            create_encoded_data()
        
        df = pd.read_csv(ENCODED_DATA_PATH)
        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки закодированных данных: {e}", exc_info=True)
        raise


def create_encoded_data():
    try:
        df = pd.read_csv(RAW_DATA_PATH, index_col=0)
        df_final = preprocess_data(df)
        df_final.to_csv(ENCODED_DATA_PATH, index=False)
    except Exception as e:
        logger.error(f"Ошибка создания закодированных данных: {e}", exc_info=True)
        raise
    

def preprocess_data(df: pd.DataFrame):
    try:
        df_encoded = encode_data(df)
        df_final = delete_unused_columns(df_encoded)
        return df_final
    except Exception as e:
        logger.error(f"Ошибка предобработки данных: {e}", exc_info=True)
        raise
    
def encode_data(df: pd.DataFrame):
    try:
        ohe = get_encoder()
        sex_encoded = ohe.transform(df[['Sex']])
        one_hot_df = pd.DataFrame(sex_encoded, columns=ohe.get_feature_names_out(['Sex']))
        df_encoded = pd.concat([df, one_hot_df], axis=1)

        return df_encoded
    except Exception as e:
        logger.error(f"Ошибка кодирования данных: {e}", exc_info=True)
        raise


def delete_unused_columns(df: pd.DataFrame):
    try:
        unused_columns = ['Sex', 'Sex_M', 'Sex_F', 'Length']
        df = df.drop(unused_columns, axis=1)
        return df
    except Exception as e:
        logger.error(f"Ошибка удаления ненужных столбцов: {e}", exc_info=True)
        raise


def get_encoder():
    try:
        if not os.path.exists(f"{MODELS_PATH}{ENCODER_FILE_NAME}"):
            create_new_encoder()
        
        with open(f"{MODELS_PATH}{ENCODER_FILE_NAME}", "rb") as f:
            ohe = pickle.load(f)
        return ohe
    except Exception as e:
        logger.error(f"Ошибка загрузки encoder: {e}", exc_info=True)
        raise


def create_new_encoder():
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(df[['Sex']])
        with open(f"{MODELS_PATH}{ENCODER_FILE_NAME}", "wb") as f:
            pickle.dump(ohe, f)
    except Exception as e:
        logger.error(f"Ошибка сохранения encoder: {e}", exc_info=True)
        raise
    


if __name__ == '__main__':
    create_encoded_data()