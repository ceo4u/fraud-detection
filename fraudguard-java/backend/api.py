from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:
    from backend.model_service import service, REPORTS_DIR
    from backend.schemas import (
        PredictionRequest,
        PredictionResponse,
        ProviderPrediction,
        ShapExplanation,
        ModelMetrics,
    )
except ImportError:
    from model_service import service, REPORTS_DIR
    from schemas import (
        PredictionRequest,
        PredictionResponse,
        ProviderPrediction,
        ShapExplanation,
        ModelMetrics,
    )

app = FastAPI(
    title="FraudGuard API",
    description="Provider-level Healthcare Fraud Detection & Behavioral Explainability Engine",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "xgboost",
        "pr_auc": float(service.metrics["PR-AUC"]),
        "threshold": float(service.threshold),
    }


@app.get("/providers", response_model=list[ProviderPrediction])
def list_providers(
    limit: int = Query(50, ge=1, le=1000),
    min_prob: float = Query(0.0, ge=0.0, le=1.0),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
):
    return service.list_predictions(limit, min_prob, sort)


@app.get("/providers/count")
def provider_count():
    total = len(service.test_preds)
    high_risk = int((service.test_preds["y_proba"] >= service.threshold).sum())
    return {
        "total": total,
        "high_risk": high_risk,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        return service.predict_single(payload.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/explain/{provider_id}", response_model=ShapExplanation)
def explain(provider_id: str):
    try:
        return service.explain(provider_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/metrics", response_model=ModelMetrics)
def metrics():
    return service.metrics_summary()


@app.get("/download-csv")
def download_csv(source: str = "test"):
    filename = "all_providers_predictions.csv" if source == "all" else "provider_fraud_predictions.csv"
    path = Path(filename)
    if not path.exists():
        path = REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(path, media_type="text/csv", filename=filename)


@app.get("/demo-provider")
def demo():
    return service.get_demo_provider()


@app.get("/plots/{name}")
def get_plot(name: str):
    allowed = {
        "shap_summary",
        "top10_importance",
        "eda_class_balance",
        "eda_distributions",
        "eda_correlation_heatmap",
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Plot not found")
    path = REPORTS_DIR / f"{name}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image {name}.png does not exist")
    return FileResponse(path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
