"""Fairness check on a SYNTHETIC group attribute.

Iris has no real sensitive attribute, so we construct one (split by median sepal
length) purely to demonstrate the group-fairness workflow: per-group accuracy and
selection rate, then the disparity. The mechanics transfer to a real attribute
(gender, age) in a credit/churn model.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib

from src.data import load_raw
from src.features import add_features

SEPAL = "sepal length (cm)"


def build_groups(df):
    median = df[SEPAL].median()                       # SYNTHETIC grouping
    return (df[SEPAL] >= median).map({True: "group_A", False: "group_B"})


def main(model_path: str = "model.joblib"):
    df = load_raw()
    df["group"] = build_groups(df)
    X = add_features(df.drop(columns=["species", "group"]))
    model = joblib.load(model_path)
    df["pred"] = model.predict(X)
    df["correct"] = (df["pred"] == df["species"]).astype(int)

    report = df.groupby("group").agg(
        n=("species", "size"),
        accuracy=("correct", "mean"),
        selection_rate_class0=("pred", lambda s: (s == 0).mean()),
    )
    print(report.round(3).to_string())
    acc_gap = report["accuracy"].max() - report["accuracy"].min()
    sel_gap = (report["selection_rate_class0"].max()
               - report["selection_rate_class0"].min())
    print(f"\nAccuracy disparity      : {acc_gap:.3f}")
    print(f"Selection-rate disparity: {sel_gap:.3f}")
    print("NOTE: 'group' is synthetic; this demonstrates the workflow only.")


if __name__ == "__main__":
    main()
