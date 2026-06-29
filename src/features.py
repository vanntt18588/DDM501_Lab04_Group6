"""Feature engineering shared by training and serving.
"""
import pandas as pd


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["petal_area"] = X["petal length (cm)"] * X["petal width (cm)"]
    X["sepal_petal_ratio"] = X["sepal length (cm)"] / (X["petal length (cm)"] + 1e-6)
    return X
