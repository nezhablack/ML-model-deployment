import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import joblib, os

def load_data(path='https://raw.githubusercontent.com/nezhablack/ML-model-deployment/refs/heads/main/data/UCI_Credit_Card.csv'):
    df = pd.read_csv(path)
    df = df.rename(columns={'default.payment.next.month': 'target'})
    df = df.drop(columns=['ID'])
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def find_best_threshold(pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.7, 0.01)
    best_t = max(thresholds, key = lambda t: f1_score(y_test, (proba >= t).astype(int)))
    return best_t

def train_and_save():
    os.makedirs('models', exist_ok = True)

    X, y = load_data()
    print(f'Размер датасета: {X.shape}')
    print(f'Баланс классов: {y.value_counts(normalize=True).round(3)}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    print(f'Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter = 1000, random_state = 42, class_weight = 'balanced'))
    ])
    pipeline.fit(X_train, y_train)

    best_t_v1 = find_best_threshold(pipeline, X_test, y_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= best_t_v1).astype(int)
    print(f'Лучший порог: {best_t_v1:.2f}')
    print(f'F1: {f1_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))

    joblib.dump({'pipeline': pipeline, 'threshold': best_t_v1}, 'models/model_v1.pkl')
    print('Модель сохранена в models/model_v1.pkl')

if __name__ == "__main__":
    train_and_save()