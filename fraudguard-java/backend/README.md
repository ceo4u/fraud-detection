# FraudGuard Backend API (FastAPI)

FastAPI inference service exposing the trained XGBoost Medicare fraud detection model, provider risk scores, SHAP explanations, and model evaluation metrics.

## Endpoints

- `GET /health` — Service health and model status
- `GET /providers?limit=50&min_prob=0.0&sort=desc` — Paginated list of provider fraud scores
- `GET /providers/count` — Total providers and high-risk flagged count
- `POST /predict` — Score a provider from a feature dictionary
- `GET /explain/{provider_id}` — SHAP attribution vectors and top 10 fraud drivers
- `GET /metrics` — PR-AUC, ROC-AUC, Precision, Recall, F1, and confusion matrix
- `GET /demo-provider` — Verified demo case (`PRV52340`) for deterministic testing
- `GET /plots/{name}` — Static PNG evaluation and EDA visualizations

## Running Standalone

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
