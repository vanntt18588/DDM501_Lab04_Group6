"""Simple data-drift demo

Compares a reference sample against a 'current' sample that we deliberately
shift, writes a full HTML report, and prints a short drift summary to the
console so you see the result immediately.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidently import Report
from evidently.presets import DataDriftPreset

from src.data import load_raw

OUT = "monitoring/drift_report.html"
FEATURES = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
]


def main():
    df = load_raw()[FEATURES]
    reference = df.sample(frac=0.5, random_state=1)
    current = df.drop(reference.index).copy()

    # Inject drift: shift petal length to simulate a changing input distribution.
    current["petal length (cm)"] = current["petal length (cm)"] + 1.5

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    snapshot.save_html(OUT)
    print("Saved drift report to", OUT)

    # --- short console summary ---
    metrics = snapshot.dict().get("metrics", [])
    per_column = []
    for m in metrics:
        name = m.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            v = m["value"]
            print(f"\nDrifted columns: {int(v['count'])} of {len(FEATURES)} "
                  f"(share {v['share']:.0%})")
        elif name.startswith("ValueDrift"):
            per_column.append((m["config"]["column"], m["value"]))

    for column, p_value in per_column:
        flag = "DRIFT" if isinstance(p_value, (int, float)) and p_value < 0.05 else "ok"
        print(f"  {column:22s} p={p_value:.3f}  {flag}")


if __name__ == "__main__":
    main()
