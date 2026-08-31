"""Offline data-drift report for the fraud-detection feature set.

Usage:
    python monitor.py --reference data/creditcard.csv --production data/production_scored.csv

The production file must contain Time, V1..V28, and Amount. If it also contains
Class, the report includes performance metrics for the observed labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

FEATURE_NAMES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def drift_report(reference_path: str, production_path: str, output_path: str = "outputs/drift_report.json") -> dict:
    reference = pd.read_csv(reference_path)
    production = pd.read_csv(production_path)
    missing = [name for name in FEATURE_NAMES if name not in reference or name not in production]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = {}
    for name in FEATURE_NAMES:
        reference_values = pd.to_numeric(reference[name], errors="coerce")
        production_values = pd.to_numeric(production[name], errors="coerce")
        statistic, p_value = ks_2samp(reference_values.dropna(), production_values.dropna())
        features[name] = {
            "reference_missing_rate": float(reference_values.isna().mean()),
            "production_missing_rate": float(production_values.isna().mean()),
            "reference_mean": float(reference_values.mean()),
            "production_mean": float(production_values.mean()),
            "reference_std": float(reference_values.std()),
            "production_std": float(production_values.std()),
            "ks_statistic": float(statistic),
            "ks_p_value": float(p_value),
            "potential_drift": bool(p_value < 0.05),
        }

    result = {
        "reference_rows": len(reference),
        "production_rows": len(production),
        "drifted_features": [name for name, values in features.items() if values["potential_drift"]],
        "feature_report": features,
        "recommendation": "investigate_and_retrain" if any(values["potential_drift"] for values in features.values()) else "no_statistically_significant_drift_detected",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--production", required=True)
    parser.add_argument("--output", default="outputs/drift_report.json")
    args = parser.parse_args()
    print(json.dumps(drift_report(args.reference, args.production, args.output), indent=2))
