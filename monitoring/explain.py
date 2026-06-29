"""Model explainability with SHAP.

The model is a Pipeline(StandardScaler, RandomForest). 
We split it: transform the features with the scaler, then run shap.TreeExplainer on the RandomForest step.
Saves a global summary plot to monitoring/shap_summary.png.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.data import load_raw
from src.features import add_features

OUT = "monitoring/shap_summary.png"


def main(model_path: str = "model.joblib"):
    X = add_features(load_raw().drop(columns=["species"]))
    pipe = joblib.load(model_path)
    scaler = pipe.named_steps["scaler"]
    rf = pipe.named_steps["model"]

    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    shap.summary_plot(shap_values, X, show=False)     # X gives feature names
    plt.tight_layout()
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved global SHAP summary to", OUT)

    pred = int(pipe.predict(X.iloc[[0]])[0])
    print(f"Row 0 predicted class: {pred}")


if __name__ == "__main__":
    main()
