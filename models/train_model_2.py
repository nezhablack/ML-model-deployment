# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from google.colab import files

model_dir = 'models'
model_path = 'models/model_v2.pkl'

# 1. Загрузка данных
df = pd.read_csv('https://raw.githubusercontent.com/nezhablack/ML-model-deployment/refs/heads/main/data/UCI_Credit_Card.csv')

df = df.rename(columns={'default.payment.next.month': 'target'})
df = df.drop(columns=['ID'])

X = df.drop(columns=['target'])
y = df['target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

def make_pipeline(class_weight_multiplier = 1.0):
    model = HistGradientBoostingClassifier(
        learning_rate = 0.05,
        max_iter = 200,
        max_depth = 6,
        min_samples_leaf = 20,
        random_state = 42
    )
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', model),
    ])
    return pipe

weight_candidates = [1.0, 2.0, 3.0, 5.0]

best_score = -1
best_weight = None
best_pipe = None
best_threshold = 0.5

for w in weight_candidates:
    pipe = make_pipeline()

    sample_weight = np.where(y_train == 1, w, 1.0)
    pipe.fit(X_train, y_train, gb__sample_weight=sample_weight)

    proba = pipe.predict_proba(X_valid)[:, 1]
    thresholds = np.arange(0.10, 0.90, 0.01)

    best_t_for_w = max(thresholds, key=lambda t: f1_score(y_valid, (proba >= t).astype(int))    )
    score = f1_score(y_valid, (proba >= best_t_for_w).astype(int))

    print(f'weight = {w}, best_t = {best_t_for_w:.2f}, F1 = {score:.4f}')

    if score > best_score:
        best_score = score
        best_weight = w
        best_pipe = pipe
        best_threshold = float(best_t_for_w)

print(f'Best weight = {best_weight}, threshold = {best_threshold:.2f}, F1 = {best_score:.4f}')

os.makedirs(model_dir, exist_ok=True)

model_obj_v2 = {
    'pipeline': best_pipe,
    'threshold': best_threshold,
    'model_version': 'v2',
}

joblib.dump(model_obj_v2, model_path)
print(f'Сохранена model_v2 в {model_path}')

"""print(sklearn.__version__)"""


