"""Train the Iris model and serialize it to model.joblib.

Uses the same model shape as Lab 02: a Pipeline(StandardScaler, RandomForest).
Feature engineering (add_features) is applied before the pipeline, matching the
serving path in api/main.py.
"""
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.data import load_raw
from src.features import add_features

SEED = 42

def train(path: str = "model.joblib", seed: int = SEED):
    df = load_raw()
    X = add_features(df.drop(columns=["species"]))
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("model", RandomForestClassifier(n_estimators=100, random_state=seed))]
    ).fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    joblib.dump(pipe, path)
    print("Saved model to", path)
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"Logging to MLflow at {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("iris_classification")
        with mlflow.start_run() as run:
            mlflow.log_metric("accuracy", acc)
            mlflow.log_param("seed", seed)
            mlflow.sklearn.log_model(pipe, "model")
            
            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "iris_model")
            print("Model registered to MLflow.")
            
    return pipe

if __name__ == "__main__":
    train()
