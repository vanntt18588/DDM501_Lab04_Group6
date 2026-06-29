"""Load the Iris dataset"""
from sklearn.datasets import load_iris


def load_raw():
    """Return Iris as a DataFrame: 4 sklearn-named feature columns + 'species'."""
    return load_iris(as_frame=True).frame.rename(columns={"target": "species"})
