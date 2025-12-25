from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from eda import preprocess_data
from model import get_model
import logging

logger = logging.getLogger(__name__)

MODELS_PATH='../models/'
MODEL_FILE_NAME='gb_model.cbm'

app = Flask(__name__)
model = GradientBoostingRegressor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json
        df = preprocess_data(pd.DataFrame(content, index=[0]))
        result = {'Rings': model.predict(df)[0]}
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route("/health")
def health():
    logger.debug("Health check")
    return jsonify({"status": "OK"}), 200


if __name__ == '__main__':
    model = get_model()
    app.run(debug=True, host='0.0.0.0')