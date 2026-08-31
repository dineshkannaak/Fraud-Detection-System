# Portfolio Post Template

I built and deployed an end-to-end fraud detection system that turns a highly imbalanced classification problem into a monitored, explainable API product.

The system combines:

- XGBoost with imbalance-aware training
- Leakage-safe preprocessing and validation
- Business-cost threshold selection
- Probability calibration
- Precision, recall, F1, PR-AUC, confusion-matrix, calibration, and error analysis
- FastAPI prediction and batch endpoints
- Optional local SHAP reasons
- Streamlit frontend
- Docker and Render deployment
- MLflow experiment tracking
- GitHub Actions quality gates
- Request IDs, API-key protection, rate controls, and Prometheus-style metrics

The API returns fraud probability, risk category, decision threshold, recommended action, model version, request ID, and optional explanation reasons. The frontend keeps the main decision simple and places supporting details behind an explanation expander.

The most important lesson was that a fraud model is not just a high offline score. It needs honest validation, calibrated probabilities, business-aware decisions, explanations, secure deployment, tests, monitoring, and a clear rollback path.

Project: [ADD_GITHUB_LINK]
Live frontend: [ADD_STREAMLIT_URL]
Live API: [ADD_API_URL]
Swagger: [ADD_API_URL]/docs
MLflow evidence: [ADD_SCREENSHOT_OR_TRACKING_URL]

#MachineLearning #FraudDetection #XGBoost #FastAPI #MLOps #ExplainableAI #Python #DataScience
