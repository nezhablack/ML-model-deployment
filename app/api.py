# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import joblib
import pandas as pd

import os

app = Flask(__name__)

MODEL_PATHS = {
    'v1': 'models/model_v1.pkl',
    'v2': 'models/model_v2.pkl',
}

FEATURE_NAMES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

MODELS = {}


def load_models():
    for version, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            continue

        model_obj = joblib.load(path)

        if isinstance(model_obj, dict):
            pipeline = model_obj['pipeline']
            threshold = float(model_obj.get('threshold', 0.5))
            model_version = model_obj.get('model_version', version)
        else:
            pipeline = model_obj
            threshold = 0.5
            model_version = version

        MODELS[version] = {
            'pipeline': pipeline,
            'threshold': threshold,
            'model_version': model_version,
        }


load_models()


def preprocess_input(data):
    if not data:
        raise ValueError("Empty JSON body")

    if "features" in data:
        features = data["features"]
        if not isinstance(features, list):
            raise ValueError("'features' must be a list")

        if len(features) != len(FEATURE_NAMES):
            raise ValueError(
                f"'features' must contain {len(FEATURE_NAMES)} values"
            )

        return pd.DataFrame([features], columns=FEATURE_NAMES)

    missing = [key for key in FEATURE_NAMES if key not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    row = [data[key] for key in FEATURE_NAMES]
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def run_prediction(data):
    model_version = data.get('model_version', 'v1')

    if model_version not in MODELS:
        raise ValueError(f'Неизвесная версия : {model_version}')

    features = preprocess_input(data)

    model_data = MODELS[model_version]
    pipeline = model_data["pipeline"]
    threshold = model_data["threshold"]

    probability = float(pipeline.predict_proba(features)[0][1])
    prediction = int(probability >= threshold)

    return {
        'model_version': model_version,
        'prediction': prediction,
        'probability': round(probability, 4),
        'threshold': round(threshold, 4)
    }


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = run_prediction(data)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'available_models': list(MODELS.keys())
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)