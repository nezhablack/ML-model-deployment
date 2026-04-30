import os
import joblib
import numpy as np

MODEL_DIR = 'models'

MODEL_PATHS = {
    'v1': os.path.join(MODEL_DIR, 'model_v1.pkl'),
    'v2': os.path.join(MODEL_DIR, 'model_v2.pkl'),
}

MODELS = {}


def load_models():
    global MODELS

    for version, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model file not found for {version}: {path}')

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


def predict(features, model_version = 'v1'):
    if model_version not in MODELS:
        raise ValueError(f'Unknown model_version: {model_version}')

    model_data = MODELS[model_version]
    pipeline = model_data['pipeline']
    threshold = model_data['threshold']

    X = np.array(features).reshape(1, -1)
    probability = float(pipeline.predict_proba(X)[0][1])
    prediction = int(probability >= threshold)

    return {
        'model_version': model_version,
        'prediction': prediction,
        'probability': round(probability, 4),
        'threshold': round(threshold, 4),
    }