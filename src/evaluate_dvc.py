"""Display metrics from a completed pipeline run for the DVC evaluate stage."""

import json
import math
from pathlib import Path

def read_metric(section: dict, name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Missing or nonnumeric metric: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite metric: {name}")
    return result

def main() -> None:
    path = Path("models/results.json")
    with path.open(encoding="utf-8") as source:
        results = json.load(source)

    regression = results.get("regression")
    if not isinstance(regression, dict):
        raise ValueError("Regression metrics are missing")
    r2 = read_metric(regression, "R2")
    mae = read_metric(regression, "MAE")
    print(f"Regression R2={r2:.4f}, MAE={mae:,.0f}")

    classification = results.get("classification")
    if classification is not None:
        if not isinstance(classification, dict):
            raise ValueError("Classification metrics have an invalid format")
        f1 = read_metric(classification, "F1")
        accuracy = read_metric(classification, "Accuracy")
        print(f"Classification F1={f1:.4f}, Accuracy={accuracy:.4f}")

if __name__ == "__main__":
    main()
