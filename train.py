"""Leakage-safe fraud detection training and evaluation pipeline.

Run:
    python train.py

Expected input:
    data/creditcard.csv

Runtime artifacts written to models/:
    fraud_model.pkl, amount_scaler.pkl, time_scaler.pkl, threshold.pkl, metadata.json
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import date
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.over_sampling import RandomOverSampler, SMOTE  # noqa: E402
from imblearn.under_sampling import RandomUnderSampler  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURE_NAMES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
FN_COST = float(os.getenv("FN_COST", "122.21"))
FP_COST = float(os.getenv("FP_COST", "5.00"))
REVIEW_COST = float(os.getenv("REVIEW_COST", "1.00"))


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    dtype_map = {f"V{i}": "float32" for i in range(1, 29)}
    dtype_map.update({"Time": "float32", "Amount": "float32", "Class": "int8"})
    df = pd.read_csv(path, dtype=dtype_map)
    required = [*FEATURE_NAMES, "Class"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[required].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns")
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before:,} rows; dropped {before - len(df):,} duplicates; "
          f"{len(df):,} remain; fraud rate={df['Class'].mean():.4%}")
    return df


def split_raw_data(df: pd.DataFrame):
    """Split raw values before fitting preprocessing: train 70%, validation 10%, calibration 10%, test 10%."""
    X_raw = df[FEATURE_NAMES].copy()
    y = df["Class"].copy()
    X_train_raw, X_holdout_raw, y_train, y_holdout = train_test_split(
        X_raw, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val_raw, X_remaining_raw, y_val, y_remaining = train_test_split(
        X_holdout_raw, y_holdout, test_size=2 / 3, random_state=RANDOM_STATE, stratify=y_holdout
    )
    X_cal_raw, X_test_raw, y_cal, y_test = train_test_split(
        X_remaining_raw, y_remaining, test_size=0.50, random_state=RANDOM_STATE, stratify=y_remaining
    )
    return X_train_raw, X_val_raw, X_cal_raw, X_test_raw, y_train, y_val, y_cal, y_test


def audit_data_leakage(*datasets: pd.DataFrame) -> dict[str, Any]:
    """Check for identical feature rows shared across any split."""
    hashes = [set(pd.util.hash_pandas_object(dataset[FEATURE_NAMES], index=False)) for dataset in datasets]
    overlap_pairs = {}
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            overlap_pairs[f"split_{left}_split_{right}"] = len(hashes[left].intersection(hashes[right]))
    result = {"duplicate_overlap_by_split_pair": overlap_pairs, "leakage_detected": any(overlap_pairs.values())}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/leakage_audit.json", "w", encoding="utf-8") as audit_file:
        json.dump(result, audit_file, indent=2)
    if result["leakage_detected"]:
        raise ValueError(f"Duplicate feature rows detected across splits: {overlap_pairs}")
    return result


def fit_preprocessors(X_train_raw: pd.DataFrame):
    """Fit scalers on training data only and transform later splits with them."""
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()
    amount_scaler.fit(X_train_raw[["Amount"]])
    time_scaler.fit(X_train_raw[["Time"]])
    return amount_scaler, time_scaler


def transform_features(X_raw: pd.DataFrame, amount_scaler: StandardScaler, time_scaler: StandardScaler) -> pd.DataFrame:
    X = X_raw.copy()
    X["Amount"] = amount_scaler.transform(X[["Amount"]]).ravel()
    X["Time"] = time_scaler.transform(X[["Time"]]).ravel()
    return X[FEATURE_NAMES]


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    return smote.fit_resample(X_train, y_train)


def make_xgb(scale_pos_weight: float, n_estimators: int = 1000, early_stopping: bool = True) -> XGBClassifier:
    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if early_stopping:
        params["early_stopping_rounds"] = 50
    return XGBClassifier(**params)


def classification_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def threshold_analysis(y_true, probabilities, thresholds: list[float] | None = None) -> pd.DataFrame:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(y_true, predictions, probabilities)
        metrics["threshold"] = float(threshold)
        metrics["fraud_predictions"] = int(predictions.sum())
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_business_threshold(y_true, probabilities) -> tuple[float, pd.DataFrame]:
    """Choose the threshold minimizing FN, FP, and manual-review costs on validation data."""
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        review_count = int(((probabilities >= 0.30) & (probabilities < threshold)).sum())
        total_cost = fn * FN_COST + fp * FP_COST + review_count * REVIEW_COST
        rows.append({"threshold": float(threshold), "cost": float(total_cost), "fn": int(fn), "fp": int(fp), "tp": int(tp), "tn": int(tn), "review_count": review_count})
    analysis = pd.DataFrame(rows)
    best = analysis.loc[analysis["cost"].idxmin(), "threshold"]
    return float(best), analysis


def train_production_model(X_train, y_train, X_val, y_val):
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / positives
    model = make_xgb(scale_pos_weight=scale_pos_weight, early_stopping=True)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, scale_pos_weight


def evaluate_model(model, X, y, threshold: float, probabilities=None) -> dict[str, Any]:
    probabilities = model.predict_proba(X)[:, 1] if probabilities is None else np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    metrics = classification_metrics(y, predictions, probabilities)
    metrics["threshold"] = float(threshold)
    metrics["classification_report"] = classification_report(y, predictions, output_dict=True, zero_division=0)
    return metrics


def fit_probability_calibrator(y_true, probabilities) -> IsotonicRegression:
    """Fit calibration only on validation predictions, never on the final test set."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities, y_true)
    return calibrator


def calibration_analysis(y_true, raw_probabilities, calibrated_probabilities, output_path: str = "outputs/calibration_curve.png") -> dict[str, float]:
    raw_brier = brier_score_loss(y_true, raw_probabilities)
    calibrated_brier = brier_score_loss(y_true, calibrated_probabilities)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 5))
    for label, probabilities in [("Raw XGBoost", raw_probabilities), ("Isotonic calibrated", calibrated_probabilities)]:
        fraction_positive, mean_predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
        plt.plot(mean_predicted, fraction_positive, marker="o", label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.title("Probability calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return {"raw_brier_score": float(raw_brier), "calibrated_brier_score": float(calibrated_brier)}


def error_analysis(X_test_raw: pd.DataFrame, y_true, probabilities, threshold: float) -> pd.DataFrame:
    """Persist a row-level error report without exposing identifiers or sensitive records."""
    predictions = (probabilities >= threshold).astype(int)
    report = X_test_raw.copy().reset_index(drop=True)
    report["actual_class"] = np.asarray(y_true).astype(int)
    report["predicted_class"] = predictions
    report["fraud_probability"] = probabilities
    report["error_type"] = np.select(
        [(report["actual_class"] == 0) & (predictions == 1), (report["actual_class"] == 1) & (predictions == 0)],
        ["false_positive", "false_negative"],
        default="correct",
    )
    report = report[report["error_type"] != "correct"].sort_values("fraud_probability", ascending=False)
    os.makedirs("outputs", exist_ok=True)
    report.to_csv("outputs/error_analysis.csv", index=False)
    summary = report.groupby("error_type", dropna=False).agg(
        count=("error_type", "size"),
        mean_amount=("Amount", "mean"),
        mean_time=("Time", "mean"),
        mean_probability=("fraud_probability", "mean"),
    ).reset_index()
    summary.to_csv("outputs/error_analysis_summary.csv", index=False)
    return report


def save_regression_fixtures(X_test: pd.DataFrame, y_true, probabilities, threshold: float) -> None:
    """Save anonymized, deterministic fixtures for prediction-regression checks."""
    fixture = X_test.head(20).copy()
    fixture["expected_probability"] = probabilities[: len(fixture)]
    fixture["expected_prediction"] = (probabilities[: len(fixture)] >= threshold).astype(int)
    fixture.to_csv("outputs/regression_fixtures.csv", index=False)


def cross_validation_summary(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float) -> dict[str, float]:
    """Use fresh fold models without early stopping so each fold is independent."""
    model = make_xgb(scale_pos_weight=scale_pos_weight, n_estimators=300, early_stopping=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "precision": "precision", "f1": "f1"},
        n_jobs=-1,
        error_score="raise",
    )
    return {f"cv_{metric}_mean": float(scores[f"test_{metric}"].mean()) for metric in ["roc_auc", "pr_auc", "recall", "precision", "f1"]} | {
        "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
        "cv_pr_auc_std": float(scores["test_pr_auc"].std()),
    }


def baseline_comparison(X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight: float) -> pd.DataFrame:
    """Compare baselines and imbalance strategies on untouched validation data."""
    X_smote, y_smote = apply_smote(X_train, y_train)
    X_ros, y_ros = RandomOverSampler(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    X_rus, y_rus = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    X_smote_rus, y_smote_rus = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X_smote, y_smote)

    experiments = {
        "DummyClassifier": (DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE), X_train, y_train),
        "LogisticRegression_balanced": (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE), X_train, y_train),
        "DecisionTree_balanced": (DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=RANDOM_STATE), X_train, y_train),
        "RandomForest_balanced": (RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE), X_train, y_train),
        "LogisticRegression_random_oversampling": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), X_ros, y_ros),
        "LogisticRegression_random_undersampling": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), X_rus, y_rus),
        "LogisticRegression_SMOTE": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), X_smote, y_smote),
        "LogisticRegression_SMOTE_undersampling": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), X_smote_rus, y_smote_rus),
        "XGBoost_scale_pos_weight": (make_xgb(scale_pos_weight=scale_pos_weight, n_estimators=300, early_stopping=False), X_train, y_train),
        "XGBoost_SMOTE": (make_xgb(scale_pos_weight=1.0, n_estimators=300, early_stopping=False), X_smote, y_smote),
    }
    rows = []
    for name, (model, features, labels) in experiments.items():
        model.fit(features, labels)
        probabilities = model.predict_proba(X_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = classification_metrics(y_val, predictions, probabilities)
        rows.append({"model": name, **{key: metrics[key] for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]}})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)


def save_artifacts(model, amount_scaler, time_scaler, threshold: float, metadata: dict[str, Any], calibrator: IsotonicRegression | None = None) -> None:
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(amount_scaler, "models/amount_scaler.pkl")
    joblib.dump(time_scaler, "models/time_scaler.pkl")
    joblib.dump(float(threshold), "models/threshold.pkl")
    if calibrator is not None:
        joblib.dump(calibrator, "models/probability_calibrator.pkl")
    with open("models/metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)


def log_to_mlflow(model, metadata: dict[str, Any]) -> str | None:
    try:
        import mlflow
        import mlflow.xgboost

        mlflow.set_experiment("fraud-detection")
        with mlflow.start_run(run_name="xgb-production-v2") as run:
            for key, value in metadata.get("parameters", {}).items():
                mlflow.log_param(key, value)
            for key, value in metadata.get("test_metrics", {}).items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            mlflow.xgboost.log_model(model, "model")
            if os.getenv("MLFLOW_REGISTER", "false").lower() in {"1", "true", "yes"}:
                registered_name = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection-xgboost")
                registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", registered_name)
                try:
                    from mlflow import MlflowClient

                    MlflowClient().set_registered_model_alias(registered_name, "candidate", registered.version)
                except Exception as alias_error:
                    print(f"MLflow alias assignment skipped: {alias_error}")
            for artifact in ["outputs/leakage_audit.json", "outputs/classification_report.json", "outputs/threshold_analysis.csv", "outputs/business_cost_curve.csv", "outputs/baseline_comparison.csv", "outputs/calibration_curve.png", "outputs/error_analysis.csv", "outputs/error_analysis_summary.csv", "outputs/regression_fixtures.csv"]:
                if os.path.exists(artifact):
                    mlflow.log_artifact(artifact)
            return run.info.run_id
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")
        return None


def main() -> None:
    started = time.perf_counter()
    df = load_data()
    X_train_raw, X_val_raw, X_cal_raw, X_test_raw, y_train, y_val, y_cal, y_test = split_raw_data(df)
    leakage_audit = audit_data_leakage(X_train_raw, X_val_raw, X_cal_raw, X_test_raw)
    amount_scaler, time_scaler = fit_preprocessors(X_train_raw)
    X_train = transform_features(X_train_raw, amount_scaler, time_scaler)
    X_val = transform_features(X_val_raw, amount_scaler, time_scaler)
    X_cal = transform_features(X_cal_raw, amount_scaler, time_scaler)
    X_test = transform_features(X_test_raw, amount_scaler, time_scaler)

    model, scale_pos_weight = train_production_model(X_train, y_train, X_val, y_val)
    val_probabilities = model.predict_proba(X_val)[:, 1]
    cal_raw_probabilities = model.predict_proba(X_cal)[:, 1]
    calibrator = fit_probability_calibrator(y_cal, cal_raw_probabilities)
    val_calibrated_probabilities = calibrator.predict(val_probabilities)
    threshold, cost_curve = select_business_threshold(y_val, val_calibrated_probabilities)
    os.makedirs("outputs", exist_ok=True)
    threshold_analysis(y_val, val_calibrated_probabilities).to_csv("outputs/threshold_analysis.csv", index=False)
    cost_curve.to_csv("outputs/business_cost_curve.csv", index=False)

    raw_test_probabilities = model.predict_proba(X_test)[:, 1]
    calibrated_test_probabilities = calibrator.predict(raw_test_probabilities)
    test_metrics = evaluate_model(model, X_test, y_test, threshold, probabilities=calibrated_test_probabilities)
    calibration_metrics = calibration_analysis(y_test, raw_test_probabilities, calibrated_test_probabilities)
    error_analysis(X_test_raw, y_test, calibrated_test_probabilities, threshold)
    save_regression_fixtures(X_test, y_test, calibrated_test_probabilities, threshold)
    with open("outputs/classification_report.json", "w", encoding="utf-8") as report_file:
        json.dump(test_metrics["classification_report"], report_file, indent=2)
    cv_metrics = cross_validation_summary(X_train, y_train, scale_pos_weight)
    baselines = baseline_comparison(X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight)
    os.makedirs("outputs", exist_ok=True)
    baselines.to_csv("outputs/baseline_comparison.csv", index=False)

    metadata = {
        "model_name": "XGBoost Fraud Detection Model",
        "model_version": os.getenv("MODEL_VERSION", "xgboost-v1.0.0"),
        "training_date": str(date.today()),
        "feature_order": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "leakage_audit": leakage_audit,
        "dataset_rows": int(len(df)),
        "legitimate_cases": int((df["Class"] == 0).sum()),
        "fraud_cases": int((df["Class"] == 1).sum()),
        "fraud_rate": float(df["Class"].mean()),
        "evaluated_thresholds": DEFAULT_THRESHOLDS,
        "imbalance_method": "scale_pos_weight",

        "threshold_selection": "minimum validation business cost after calibration",
        "probability_calibration": "isotonic_regression",
        "parameters": {
            "random_state": RANDOM_STATE,
            "train_fraction": 0.70,
            "validation_fraction": 0.10,
            "calibration_fraction": 0.10,
            "test_fraction": 0.10,
            "scale_pos_weight": scale_pos_weight,
            "decision_threshold": threshold,
            "fn_cost": FN_COST,
            "fp_cost": FP_COST,
            "review_cost": REVIEW_COST,
        },
        "test_metrics": {key: value for key, value in test_metrics.items() if isinstance(value, (int, float))},
        "calibration_metrics": calibration_metrics,
        "artifacts": [
            "fraud_model.pkl", "amount_scaler.pkl", "time_scaler.pkl",
            "probability_calibrator.pkl", "threshold.pkl", "metadata.json",
        ],
        "cross_validation": cv_metrics,
        "training_seconds": time.perf_counter() - started,
        "python_version": os.sys.version,
        "mlflow_model_name": os.getenv("MLFLOW_MODEL_NAME", "fraud-detection-xgboost"),
        "mlflow_registration_enabled": os.getenv("MLFLOW_REGISTER", "false").lower() in {"1", "true", "yes"},
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    save_artifacts(model, amount_scaler, time_scaler, threshold, metadata, calibrator=calibrator)
    run_id = log_to_mlflow(model, metadata)
    metadata["mlflow_run_id"] = run_id
    with open("models/metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)

    print(json.dumps({"threshold": threshold, "test_metrics": metadata["test_metrics"], "cv": cv_metrics, "mlflow_run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
