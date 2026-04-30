# ML model deployment

Это проект разработки ML‑сервиса для прогнозирования дефолта по кредитным картам на основе датасета Default of Credit Card Clients. Сервис покрывает: обучение модели, сохранение, REST API, Docker‑образ и концепцию A/B‑тестирования.

## 1. Цель и постановка задачи

**Цель:** построить и задеплоить веб‑сервис, который по признакам клиента кредитной карты возвращает прогноз дефолта в следующем месяце и его вероятность.

- **Предметная область:** финансы / кредитный скоринг.
- **Целевая переменная:** `default.payment.next.month` переименована в `target`.
- **Датасет:** [Default of Credit Card Clients (Kaggle)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).

## 2. Данные

- Источник: Kaggle.
- Объекты: клиенты с демографией, историей платежей и остатками по счетам.
- Таргет: бинарный признак дефолта в следующем месяце.

При подготовке:

- столбец `default.payment.next.month` переименовывается в `target`;
- столбец `ID` удаляется;
- остальные признаки используются как входные фичи.

## 3. Архитектура сервиса

Используется **монолитный Flask‑сервис**:

- при старте загружает модель из `models/model_v1.pkl`;
- предоставляет два эндпоинта:
  - `GET /health` - health‑check;
  - `POST /predict` - инференс по одному клиенту.

Такой монолитный вариант выбран, потому что в учебном проекте достаточно одной модели и одного сервиса; он проще в запуске и проверке, но при необходимости модельный слой можно выделить в отдельный сервис.

## 4. Модель и MLOps‑концепты

### 4.1. Модель

- Используется простая модель бинарной классификации `LogisticRegression` из `scikit-learn`.
- Обученная модель сохранена с помощью `joblib` в файл `models/model_v1.pkl`.
- Загрузка модели и инференс реализованы в модуле `app/model_handler.py`.

### 4.2. DVC и MLflow (концепт)

В проекте DVC и MLflow не подключены, но в полном MLOps‑цикле они могли бы использоваться:

- **DVC** для версионирования датасета и артефактов модели, связь “код - данные - модель”.
- **MLflow** для логирования экспериментов, сравнение моделей и регистрации лучшей модели для продакшена и A/B‑тестирования.

## 5. API

### 5.1. Endpoints

- `GET /health` - проверка работоспособности сервиса.
- `POST /predict` - инференс по одному клиенту.

### 5.2. Health‑check

Пример (PowerShell):

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:5000/health" -Method GET
```
Пример (curl):

curl http://127.0.0.1:5000/health

Пример ответа:

```json
{"status": "healthy"}
```

### 5.3. Predict

**Формат запроса:**

```http
POST /predict
Content-Type: application/json

{
  "features": [f1, f2, ..., fN]
}
```

`features` - список чисел в том же порядке, в котором признаки подавались модели при обучении.


**Пример запроса (PowerShell):**

```powershell
$body = @{
    features = @(20000,2,2,1,24,2,2,0,0,0,0,3913,3102,689,0,0,0,0,689,0,0,0,0)
} | ConvertTo-Json

(Invoke-WebRequest `
  -Uri "http://127.0.0.1:5000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body).Content
```
Пример запроса (curl):   
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [20000,2,2,1,24,2,2,0,0,0,0,3913,3102,689,0,0,0,0,689,0,0,0,0]}'

**Пример ответа:**

```json
{
  "prediction": 1,
  "probability": 0.7923,
  "threshold": 0.6
}
```

## 6. Запуск локально

```bash
# (опционально) создать и активировать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate  # для Windows

# установить зависимости
pip install -r requirements.txt

# запустить сервис
python -m flask --app app/api.py run
```

Сервис будет доступен по адресу `http://127.0.0.1:5000`.

## 7. Docker

### 7.1. Локальная сборка и запуск

```bash
docker build -t credit-default-api:latest .
docker run -p 5000:5000 credit-default-api:latest
```

После этого API доступен по адресу `http://127.0.0.1:5000`.

### 7.2. Образ в Docker Hub

Публичный образ:

```text
https://hub.docker.com/r/nezhablack/credit-default-api
```

Запуск без локальной сборки:

```bash
docker pull nezhablack/credit-default-api:latest
docker run -p 5000:5000 nezhablack/credit-default-api:latest
```

## 8. Бизнес‑метрики и A/B‑тестирование

Для оценки качества модели, кроме стандартных ML‑метрик (F1, Precision, Recall), учитываются бизнес‑метрики:

- **Expected Loss (ожидаемые потери)** - оценка потерь портфеля с учетом вероятностей дефолта.
- **Approval rate при фиксированном уровне риска** - доля одобренных заявок при заданном допустимом уровне дефолтов.

Предусмотрена концепция A/B‑теста двух версий модели (v1 и v2). Детальный план A/B‑теста описан в файле [ab_test_plan.md](./ab_test_plan.md).
