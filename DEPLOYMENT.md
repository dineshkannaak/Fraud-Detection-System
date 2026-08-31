# Deployment Guide

## 1. Prepare the model artifacts

Install dependencies and place the original dataset at `data/creditcard.csv`:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python train.py
```

Confirm that these files exist:

```text
models/fraud_model.pkl
models/amount_scaler.pkl
models/time_scaler.pkl
models/probability_calibrator.pkl
models/threshold.pkl
models/metadata.json
```

The API image can include these files during a private deployment, or they can be uploaded as a ZIP to private HTTPS object storage. If using object storage, the ZIP must contain the files directly and must be accompanied by its SHA-256 digest.

## 2. Local verification

```bash
python -m compileall -q app.py train.py streamlit_app.py monitor.py load_test.py tests
pytest tests/ -v --tb=short
uvicorn app:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
FRAUD_API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

Verify:

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl http://localhost:8000/metrics
```

## 3. API security configuration

For any public deployment, set a long random `API_KEY` and use the same value in the Streamlit service. Set `CORS_ORIGINS` to the exact frontend origin; do not leave it as `*`. Keep `ENABLE_SHAP=false` until the feature explanation path has been reviewed for the target data and latency budget.

| Variable | Required | Example |
|---|---:|---|
| `API_KEY` | Public deployment | long random secret |
| `CORS_ORIGINS` | Public deployment | `https://fraud-streamlit.onrender.com` |
| `FRAUD_API_URL` | Streamlit service | `https://fraud-api.onrender.com` |
| `MODEL_ARTIFACT_URL` | If artifacts are not in image | private HTTPS ZIP URL |
| `MODEL_ARTIFACT_SHA256` | With artifact URL | lowercase SHA-256 digest |
| `MODEL_ARTIFACT_TOKEN` | Private storage only | secret bearer token |
| `MODEL_VERSION` | Recommended | `xgboost-v1.0.0` |
| `RATE_LIMIT_PER_MINUTE` | Recommended | `120` |
| `MAX_REQUEST_BODY_BYTES` | Recommended | `1000000` |

## 4. Docker deployment

Build the API after training and artifact generation:

```bash
docker build -f Dockerfile -t fraud-api:latest .
docker run --rm -p 8000:8000 \
  -e API_KEY='replace-me' \
  -e CORS_ORIGINS='http://localhost:8501' \
  fraud-api:latest
```

Build and run the frontend:

```bash
docker build -f Dockerfile.streamlit -t fraud-streamlit:latest .
docker run --rm -p 8501:8501 \
  -e FRAUD_API_URL='http://host.docker.internal:8000' \
  -e API_KEY='replace-me' \
  fraud-streamlit:latest
```

## 5. Render deployment

Push this repository to GitHub and create services from `render.yaml`, or create two Docker web services manually.

For `fraud-api`, set the API key, exact frontend CORS origin, model version, and either commit private model files through a secure artifact workflow or set the three artifact-fetch variables. For `fraud-streamlit`, set `FRAUD_API_URL` to the deployed API URL and use the same API key.

After both services deploy, confirm `/health`, `/docs`, `/model-info`, and `/metrics` on the API. Open the Streamlit URL and submit a known-good test payload. Do not treat a successful health check as proof that the model is loaded; require `model_loaded: true`.

## 6. Monitoring and drift checks

The API exposes Prometheus-compatible text at `/metrics`. Scrape it with a monitoring service and alert on downtime, error rate, latency, and sudden changes in fraud-flag rate. When labels become available, calculate precision, recall, PR-AUC, false-positive rate, and false-negative rate by time period and important segments.

Run the offline drift report against the reference training data and a production export:

```bash
python monitor.py --reference data/creditcard.csv --production data/production_scored.csv
```

Investigate statistically significant drift rather than automatically retraining from unreviewed data.

## 7. Operational limitations

The included counters and rate limiter are process-local. For multiple API replicas, replace them with a shared metrics backend and distributed rate limiter. The model artifact URL should point to immutable versioned storage. Keep the previous model available for rollback. Recommendations are advisory and should feed a human-review process rather than silently making irreversible financial decisions.
