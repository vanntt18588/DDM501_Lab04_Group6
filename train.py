"""Train the Iris model and serialize it to model.joblib.

Uses the same model shape as Lab 02: a Pipeline(StandardScaler, RandomForest).
Feature engineering (add_features) is applied before the pipeline, matching the
serving path in api/main.py.
"""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_raw
from src.features import add_features

SEED = 42


def train(path: str = "model.joblib", seed: int = SEED):
    df = load_raw()
    X = add_features(df.drop(columns=["species"]))
    y = df["species"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("model", RandomForestClassifier(n_estimators=100, random_state=seed))]
    ).fit(X_train, y_train)
    joblib.dump(pipe, path)
    print("Saved model to", path)
    return pipe


if __name__ == "__main__":
    train()
