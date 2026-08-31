"""Fraud Detection API.

The original /predict request contract remains compatible:
features are ordered as [Time, V1..V28, Amount] and exactly 30 values are required.
New response fields are additive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
AMT_SCALER_PATH = os.path.join(MODEL_DIR, "amount_scaler.pkl")
TIME_SCALER_PATH = os.path.join(MODEL_DIR, "time_scaler.pkl")
THRESH_PATH = os.path.join(MODEL_DIR, "threshold.pkl")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "probability_calibrator.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

MODEL_VERSION = os.getenv("MODEL_VERSION", "xgboost-v1.0.0")
LOW_RISK_THRESHOLD = float(os.getenv("LOW_RISK_THRESHOLD", "0.30"))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.70"))
DEFAULT_DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.50"))
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "false").lower() in {"1", "true", "yes"}
PREDICTION_LOGGING = os.getenv("PREDICTION_LOGGING", "false").lower() in {"1", "true", "yes"}
API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", "1000000"))
SYNC_THREAD_TOKENS = int(os.getenv("SYNC_THREAD_TOKENS", "100"))
MAX_INFLIGHT_REQUESTS = int(os.getenv("MAX_INFLIGHT_REQUESTS", "100"))

if not 0 <= LOW_RISK_THRESHOLD < HIGH_RISK_THRESHOLD <= 1:
    raise ValueError("LOW_RISK_THRESHOLD and HIGH_RISK_THRESHOLD must satisfy 0 <= low < high <= 1")
if not 0 < DEFAULT_DECISION_THRESHOLD < 1:
    raise ValueError("DECISION_THRESHOLD must be between 0 and 1")

FEATURE_NAMES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
FEATURE_DESCRIPTIONS = {
    "Time": "Elapsed transaction time from the source dataset",
    "Amount": "Transaction amount",
    **{f"V{i}": f"Anonymized PCA feature V{i}" for i in range(1, 29)},
}
ACTIONS = {
    "low": "approve_automatically",
    "medium": "request_additional_verification",
    "high": "hold_or_manual_review",
}

model_store: dict[str, Any] = {}
request_counts: dict[str, int] = {}
latency_total_ms = 0.0
error_count = 0
overload_count = 0
rate_windows: dict[str, deque[float]] = {}
inflight_guard = threading.BoundedSemaphore(max(1, MAX_INFLIGHT_REQUESTS))
prediction_executor: ThreadPoolExecutor | None = None


def get_prediction_executor() -> ThreadPoolExecutor:
    global prediction_executor
    if prediction_executor is None or getattr(prediction_executor, "_shutdown", False):
        prediction_executor = ThreadPoolExecutor(max_workers=max(1, SYNC_THREAD_TOKENS), thread_name_prefix="fraud-predict")
    return prediction_executor


class RequestLogger(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]):
        kwargs.setdefault("extra", {}).setdefault("request_id", self.extra["request_id"])
        return msg, kwargs


def request_logger(request_id: str) -> RequestLogger:
    return RequestLogger(logger, {"request_id": request_id})


def load_artifacts() -> None:
    """Load runtime artifacts once; keep the service available in degraded mode if absent."""
    required = [MODEL_PATH, AMT_SCALER_PATH, TIME_SCALER_PATH, THRESH_PATH]
    missing = [path for path in required if not os.path.exists(path)]
    model_store.clear()
    model_store["request_count"] = 0
    model_store["fraud_count"] = 0
    model_store["threshold"] = DEFAULT_DECISION_THRESHOLD

    if missing:
        model_store["model_loaded"] = False
        logger.error("Model artifacts unavailable: %s", ", ".join(missing), extra={"request_id": "startup"})
        return

    model_store["model"] = joblib.load(MODEL_PATH)
    model_store["amount_scaler"] = joblib.load(AMT_SCALER_PATH)
    model_store["time_scaler"] = joblib.load(TIME_SCALER_PATH)
    model_store["threshold"] = float(joblib.load(THRESH_PATH))
    if os.path.exists(CALIBRATOR_PATH):
        model_store["calibrator"] = joblib.load(CALIBRATOR_PATH)
    model_store["model_loaded"] = True

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
                model_store["metadata"] = json.load(metadata_file)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read model metadata: %s", exc, extra={"request_id": "startup"})

    logger.info(
        "Fraud model loaded; threshold=%.4f; version=%s",
        model_store["threshold"],
        MODEL_VERSION,
        extra={"request_id": "startup"},
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_prediction_executor()
    try:
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = max(1, SYNC_THREAD_TOKENS)
    except Exception as exc:
        logger.warning("Could not configure sync thread capacity: %s", exc, extra={"request_id": "startup"})
    load_artifacts()
    yield
    model_store.clear()
    global prediction_executor
    if prediction_executor is not None:
        prediction_executor.shutdown(wait=True, cancel_futures=True)
        prediction_executor = None
    logger.info("Model store cleared", extra={"request_id": "shutdown"})


app = FastAPI(
    title="Fraud Detection API",
    default_response_class=ORJSONResponse,
    description="Validated real-time fraud scoring with configurable risk actions.",
    version="1.1.0",
    lifespan=lifespan,
)

allowed_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "The request contains invalid or missing fields",
            "details": jsonable_encoder(exc.errors()),
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.middleware("http")
async def add_request_id_and_controls(request: Request, call_next):
    global latency_total_ms, error_count, overload_count
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    client_id = request.client.host if request.client else "unknown"

    if request.headers.get("content-length") and int(request.headers["content-length"]) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": "request_too_large", "message": "Request body exceeds the configured limit", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
    public_paths = {"/", "/health", "/docs", "/openapi.json"}
    if API_KEY and request.url.path not in public_paths and request.headers.get("X-API-Key") != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"error": "authentication_required", "message": "A valid X-API-Key is required", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    acquired = inflight_guard.acquire(blocking=False)
    if not acquired:
        overload_count += 1
        return JSONResponse(
            status_code=429,
            content={"error": "server_overloaded", "message": "Server is at its in-flight request limit; retry later", "request_id": request_id},
            headers={"X-Request-ID": request_id, "Retry-After": "1"},
        )

    now = time.time()
    if RATE_LIMIT_PER_MINUTE > 0:
        recent = rate_windows.setdefault(client_id, deque())
        while recent and now - recent[0] >= 60:
            recent.popleft()
        if len(recent) >= RATE_LIMIT_PER_MINUTE:
            inflight_guard.release()
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "message": "Too many requests; retry later", "request_id": request_id},
                headers={"X-Request-ID": request_id, "Retry-After": "60"},
            )
        recent.append(now)

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        error_count += 1
        request_logger(request_id).exception("Unhandled request error")
        raise
    finally:
        latency_total_ms += (time.perf_counter() - started) * 1000
        inflight_guard.release()
    request_counts[request.method] = request_counts.get(request.method, 0) + 1
    response.headers["X-Request-ID"] = request_id
    return response


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(..., min_length=1, max_length=128)
    features: list[float] = Field(..., min_length=30, max_length=30)

    @field_validator("features")
    @classmethod
    def validate_features(cls, values: list[float]) -> list[float]:
        if len(values) != 30:
            raise ValueError("features must contain exactly 30 values: Time, V1-V28, Amount")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("features must contain finite numeric values")
        if values[-1] < 0:
            raise ValueError("Amount must be greater than or equal to zero")
        return values


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    prediction: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str
    decision_threshold: float = Field(..., ge=0, le=1)
    recommended_action: str
    model_version: str
    explanation_available: bool
    top_reasons: list[dict[str, Any]] = Field(default_factory=list)
    decision_ms: float = Field(..., ge=0)
    request_id: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
    request_id: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str
    threshold: float
    requests_served: int
    fraud_flagged: int


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    framework: str
    imbalance_method: str
    features: int
    feature_order: list[str]
    decision_threshold: float
    low_risk_threshold: float
    high_risk_threshold: float
    calibration_method: str
    explanation_enabled: bool
    monitoring_endpoint: str
    training_date: str | None = None


def get_decision_threshold() -> float:
    return float(model_store.get("threshold", DEFAULT_DECISION_THRESHOLD))


def get_risk_level(probability: float) -> str:
    if probability < LOW_RISK_THRESHOLD:
        return "low"
    if probability < HIGH_RISK_THRESHOLD:
        return "medium"
    return "high"


def recommended_action(risk_level: str) -> str:
    return ACTIONS[risk_level]


def preprocess_transactions(features_batch: list[list[float]]) -> np.ndarray:
    if not model_store.get("model_loaded") or "model" not in model_store:
        raise RuntimeError("Model is not loaded")

    values = np.asarray(features_batch, dtype=np.float32).reshape(-1, 30)
    values[:, 0] = model_store["time_scaler"].transform(values[:, [0]]).ravel()
    values[:, 29] = model_store["amount_scaler"].transform(values[:, [29]]).ravel()
    return values


def preprocess_transaction(features: list[float]) -> np.ndarray:
    return preprocess_transactions([features])


def local_explanation(X: np.ndarray, request_id: str) -> list[dict[str, Any]]:
    """Return best-effort local SHAP reasons without exposing raw sensitive values."""
    if not ENABLE_SHAP or "model" not in model_store:
        return []
    try:
        import shap

        values = shap.TreeExplainer(model_store["model"]).shap_values(X)
        if isinstance(values, list):
            values = values[-1]
        row = np.asarray(values)[0]
        top_indices = np.argsort(np.abs(row))[::-1][:3]
        return [
            {
                "feature": FEATURE_NAMES[int(index)],
                "description": FEATURE_DESCRIPTIONS[FEATURE_NAMES[int(index)]],
                "impact": "increased_risk" if row[index] >= 0 else "decreased_risk",
                "importance": round(float(abs(row[index])), 6),
            }
            for index in top_indices
        ]
    except Exception as exc:  # Explanation must never break a prediction.
        request_logger(request_id).warning("SHAP explanation unavailable: %s", exc)
        return []


def score_transaction(tx: TransactionRequest, request_id: str) -> PredictionResponse:
    start_time = time.perf_counter()
    request_log = request_logger(request_id)
    try:
        X = preprocess_transaction(tx.features)
        probability = float(model_store["model"].predict_proba(X)[0, 1])
        probability = min(max(probability, 0.0), 1.0)
        if "calibrator" in model_store:
            probability = float(model_store["calibrator"].predict([probability])[0])
            probability = min(max(probability, 0.0), 1.0)
        threshold = get_decision_threshold()
        is_fraud = probability >= threshold
        risk = get_risk_level(probability)
        model_store["request_count"] = int(model_store.get("request_count", 0)) + 1
        if is_fraud:
            model_store["fraud_count"] = int(model_store.get("fraud_count", 0)) + 1
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if PREDICTION_LOGGING:
            request_log.info(
                "prediction completed; fraud=%s; risk=%s; probability=%.4f; latency_ms=%.2f",
                is_fraud,
                risk,
                probability,
                elapsed_ms,
            )
        return PredictionResponse(
            transaction_id=tx.transaction_id,
            prediction="fraud" if is_fraud else "not_fraud",
            fraud_probability=round(probability, 4),
            is_fraud=is_fraud,
            risk_level=risk,
            decision_threshold=round(threshold, 4),
            recommended_action=recommended_action(risk),
            model_version=MODEL_VERSION,
            explanation_available=bool(ENABLE_SHAP and "model" in model_store),
            top_reasons=local_explanation(X, request_id),
            decision_ms=round(elapsed_ms, 2),
            request_id=request_id,
        )
    except RuntimeError as exc:
        request_log.error("prediction unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "message": "The fraud model is not ready", "request_id": request_id},
        ) from exc
    except Exception as exc:
        request_log.exception("prediction failed")
        raise HTTPException(
            status_code=500,
            detail={"error": "prediction_error", "message": "Prediction could not be completed", "request_id": request_id},
        ) from exc


def score_transactions_batch(transactions: list[TransactionRequest], request_id: str) -> list[PredictionResponse]:
    """Vectorized batch scoring to avoid repeated scaler and model calls."""
    started = time.perf_counter()
    try:
        features = [transaction.features for transaction in transactions]
        X = preprocess_transactions(features)
        probabilities = np.asarray(model_store["model"].predict_proba(X)[:, 1], dtype=float)
        if "calibrator" in model_store:
            probabilities = np.asarray(model_store["calibrator"].predict(probabilities), dtype=float)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        threshold = get_decision_threshold()
        flags = probabilities >= threshold
        risks = [get_risk_level(float(probability)) for probability in probabilities]
        model_store["request_count"] = int(model_store.get("request_count", 0)) + len(transactions)
        model_store["fraud_count"] = int(model_store.get("fraud_count", 0)) + int(flags.sum())
        elapsed_ms = (time.perf_counter() - started) * 1000
        if PREDICTION_LOGGING:
            request_logger(request_id).info("batch prediction completed; count=%d; latency_ms=%.2f", len(transactions), elapsed_ms)
        per_item_ms = elapsed_ms / max(len(transactions), 1)
        return [
            PredictionResponse(
                transaction_id=transaction.transaction_id,
                prediction="fraud" if bool(flags[index]) else "not_fraud",
                fraud_probability=round(float(probabilities[index]), 4),
                is_fraud=bool(flags[index]),
                risk_level=risks[index],
                decision_threshold=round(threshold, 4),
                recommended_action=recommended_action(risks[index]),
                model_version=MODEL_VERSION,
                explanation_available=False,
                top_reasons=[],
                decision_ms=round(per_item_ms, 2),
                request_id=request_id,
            )
            for index, transaction in enumerate(transactions)
        ]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"error": "model_unavailable", "message": "The fraud model is not ready", "request_id": request_id}) from exc
    except Exception as exc:
        request_logger(request_id).exception("batch prediction failed")
        raise HTTPException(status_code=500, detail={"error": "prediction_error", "message": "Batch prediction could not be completed", "request_id": request_id}) from exc


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(tx: TransactionRequest, request: Request):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_prediction_executor(), score_transaction, tx, request.state.request_id)


@app.post("/predict-batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(payload: BatchRequest, request: Request):
    request_id = request.state.request_id
    loop = asyncio.get_running_loop()
    predictions = await loop.run_in_executor(get_prediction_executor(), score_transactions_batch, payload.transactions, request_id)
    return BatchResponse(count=len(predictions), predictions=predictions, request_id=request_id)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    loaded = bool(model_store.get("model_loaded") and "model" in model_store)
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
        threshold=round(get_decision_threshold(), 4),
        requests_served=int(model_store.get("request_count", 0)),
        fraud_flagged=int(model_store.get("fraud_count", 0)),
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
def metrics():
    total_requests = sum(request_counts.values())
    average_latency = latency_total_ms / total_requests if total_requests else 0.0
    lines = [
        "# HELP fraud_api_requests_total Total HTTP requests received",
        "# TYPE fraud_api_requests_total counter",
        f"fraud_api_requests_total {total_requests}",
        "# HELP fraud_api_errors_total Total unhandled HTTP/application errors",
        "# TYPE fraud_api_errors_total counter",
        f"fraud_api_errors_total {error_count}",
        "# HELP fraud_api_overload_total Requests rejected by in-flight admission control",
        "# TYPE fraud_api_overload_total counter",
        f"fraud_api_overload_total {overload_count}",
        "# HELP fraud_api_predictions_total Total predictions served",
        "# TYPE fraud_api_predictions_total counter",
        f"fraud_api_predictions_total {model_store.get('request_count', 0)}",
        "# HELP fraud_api_fraud_flagged_total Total transactions classified as fraud",
        "# TYPE fraud_api_fraud_flagged_total counter",
        f"fraud_api_fraud_flagged_total {model_store.get('fraud_count', 0)}",
        "# HELP fraud_api_latency_average_ms Average middleware latency in milliseconds",
        "# TYPE fraud_api_latency_average_ms gauge",
        f"fraud_api_latency_average_ms {average_latency:.4f}",
    ]
    return PlainTextResponse("\\n".join(lines) + "\\n")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Information"])
def model_info():
    metadata = model_store.get("metadata", {})
    return ModelInfoResponse(
        model_name=metadata.get("model_name", "XGBoost Fraud Detection Model"),
        model_version=MODEL_VERSION,
        framework="XGBoost",
        imbalance_method=metadata.get("imbalance_method", "scale_pos_weight"),
        features=len(FEATURE_NAMES),
        feature_order=FEATURE_NAMES,
        decision_threshold=round(get_decision_threshold(), 4),
        low_risk_threshold=LOW_RISK_THRESHOLD,
        high_risk_threshold=HIGH_RISK_THRESHOLD,
        calibration_method=metadata.get("probability_calibration", "none"),
        explanation_enabled=ENABLE_SHAP,
        monitoring_endpoint="/metrics",
        training_date=metadata.get("training_date"),
    )


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Fraud Detection API",
        "status": "running",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "batch_predict": "/predict-batch",
    }
