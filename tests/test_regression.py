from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "outputs" / "regression_fixtures.csv"
MODEL = ROOT / "models" / "fraud_model.pkl"
AMOUNT_SCALER = ROOT / "models" / "amount_scaler.pkl"
TIME_SCALER = ROOT / "models" / "time_scaler.pkl"


@pytest.mark.skipif(not all(path.exists() for path in [FIXTURES, MODEL, AMOUNT_SCALER, TIME_SCALER]), reason="training artifacts are not present")
def test_saved_predictions_do_not_regress():
    fixture = pd.read_csv(FIXTURES)
    feature_names = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    X = fixture[feature_names].copy()
    X["Amount"] = joblib.load(AMOUNT_SCALER).transform(X[["Amount"]]).ravel()
    X["Time"] = joblib.load(TIME_SCALER).transform(X[["Time"]]).ravel()
    probabilities = joblib.load(MODEL).predict_proba(X)[:, 1]
    assert np.allclose(probabilities, fixture["expected_probability"].to_numpy(), atol=1e-5)
