"""Stress-test wrapper for the real FastAPI application.

This module patches only artifact loading/model inference. The actual app.py
middleware, validation, routes, response serialization, rate controls, and
metrics endpoint are exercised over HTTP.
"""

from __future__ import annotations

import numpy as np

import app as fraud_app


class FakeScaler:
    def transform(self, values):
        return np.asarray(values, dtype=np.float32)


class FakeModel:
    def predict_proba(self, values):
        # Deterministic vectorized inference so client/server overhead is measured
        # without depending on absent production model binaries.
        values = np.asarray(values, dtype=np.float32)
        scores = 1.0 / (1.0 + np.exp(-values[:, 29]))
        return np.column_stack((1.0 - scores, scores)).astype(np.float32)


def load_mock_artifacts() -> None:
    fraud_app.model_store.clear()
    fraud_app.model_store.update(
        {
            "model": FakeModel(),
            "amount_scaler": FakeScaler(),
            "time_scaler": FakeScaler(),
            "threshold": 0.50,
            "model_loaded": True,
            "request_count": 0,
            "fraud_count": 0,
        }
    )


fraud_app.load_artifacts = load_mock_artifacts
app = fraud_app.app
