"""Instrumented FastAPI service

Prometheus metrics:
  - standard HTTP metrics (request count, latency) via the instrumentator
  - a custom counter: predictions broken down by predicted class
exposed at GET /metrics.
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from src.features import add_features

MODEL_URI = os.environ.get("MODEL_URI")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")

app = FastAPI(title="Iris Classifier")
_model = None


def get_model():
    global _model
    if _model is None:
        if MODEL_URI:
            import mlflow.pyfunc
            try:
                _model = mlflow.pyfunc.load_model(MODEL_URI)
                print(f"Loaded model from MLflow: {MODEL_URI}")
            except Exception as e:
                print(f"Failed to load from MLflow: {e}. Falling back to local model.")
                _model = joblib.load(MODEL_PATH)
        else:
            _model = joblib.load(MODEL_PATH)
    return _model


PREDICTIONS = Counter(
    "iris_predictions_total", "Total predictions by class", ["predicted_class"]
)


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    X = pd.DataFrame([{
        "sepal length (cm)": features.sepal_length,
        "sepal width (cm)": features.sepal_width,
        "petal length (cm)": features.petal_length,
        "petal width (cm)": features.petal_width,
    }])
    X = add_features(X)
    prediction = int(get_model().predict(X)[0])
    PREDICTIONS.labels(predicted_class=str(prediction)).inc()
    return {"prediction": prediction}


Instrumentator().instrument(app).expose(app)
