"""
Model service for FraudGuard API.
Loads trained XGBoost model artifacts, features, test predictions, and SHAP data.
Provides prediction, explanation, and evaluation metrics APIs.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
CANDIDATES_MODELS = [
    CURRENT_DIR.parent.parent / "models",
    CURRENT_DIR.parent.parent / "fraud-detector" / "models",
    CURRENT_DIR.parent / "models",
]
MODELS_DIR = next((p for p in CANDIDATES_MODELS if p.exists()), CANDIDATES_MODELS[0])

CANDIDATES_REPORTS = [
    CURRENT_DIR.parent.parent / "reports",
    CURRENT_DIR.parent.parent / "fraud-detector" / "reports",
    CURRENT_DIR.parent / "reports",
]
REPORTS_DIR = next((p for p in CANDIDATES_REPORTS if p.exists()), CANDIDATES_REPORTS[0])


class ModelService:
    def __init__(self):
        print(f"[INFO] Initializing ModelService with models from {MODELS_DIR}...")
        self.model_data = joblib.load(MODELS_DIR / "model.pkl")
        self.model = self.model_data["model"]
        self.features = self.model_data["features"]
        self.threshold = float(self.model_data["threshold"])
        self.metrics = self.model_data["metrics"]

        self.features_df = pd.read_parquet(MODELS_DIR / "features.parquet")
        self.test_preds = pd.read_parquet(MODELS_DIR / "test_preds.parquet")
        self.shap_values = np.load(MODELS_DIR / "shap_values.npy")
        self.shap_X = pd.read_parquet(MODELS_DIR / "shap_X_sample.parquet")
        self.demo_provider = json.loads((MODELS_DIR / "demo_provider.json").read_text(encoding="utf-8"))
        self._shap_index = {pid: i for i, pid in enumerate(self.shap_X.index)}
        print(f"[INFO] ModelService initialized successfully. Features: {len(self.features)}, Test set: {len(self.test_preds)}")

    def predict_single(self, features_dict: dict) -> dict:
        X = pd.DataFrame([features_dict])[self.features].fillna(0)
        proba = float(self.model.predict_proba(X)[0][1])
        tier = "High" if proba >= self.threshold else ("Medium" if proba >= self.threshold * 0.5 else "Low")
        return {
            "provider": str(features_dict.get("Provider", "NEW")),
            "fraud_probability": round(proba, 4),
            "risk_tier": tier,
            "threshold": round(self.threshold, 4),
        }

    def list_predictions(self, limit: int = 50, min_prob: float = 0.0, sort: str = "desc") -> list:
        df = self.test_preds.copy()
        df = df[df["y_proba"] >= min_prob]
        df = df.sort_values("y_proba", ascending=(sort == "asc")).head(limit)
        records = df.to_dict(orient="records")
        for r in records:
            r["y_proba"] = float(r["y_proba"])
            r["y_true"] = int(r["y_true"])
            r["y_pred"] = int(r["y_pred"])
            r["risk_tier"] = "High" if r["y_proba"] >= self.threshold else ("Medium" if r["y_proba"] >= self.threshold * 0.5 else "Low")
        return records

    def explain(self, provider_id: str) -> dict:
        if provider_id not in self._shap_index:
            if provider_id in self.features_df.index:
                x_row = self.features_df.loc[provider_id, self.features]
                import shap
                explainer = shap.TreeExplainer(self.model)
                raw_shap = explainer.shap_values(x_row.values.reshape(1, -1))
                shap_row = raw_shap[0] if isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2 else np.array(raw_shap).flatten()
            else:
                demo_pid = self.demo_provider.get("provider", self.shap_X.index[0])
                idx = self._shap_index.get(demo_pid, 0)
                shap_row = self.shap_values[idx]
                x_row = self.shap_X.iloc[idx]
                provider_id = demo_pid
        else:
            idx = self._shap_index[provider_id]
            shap_row = self.shap_values[idx]
            x_row = self.shap_X.iloc[idx]

        drivers = [
            {"feature": f, "value": float(x_row[f]), "shap": float(shap_row[i])}
            for i, f in enumerate(self.features)
        ]
        drivers.sort(key=lambda d: abs(d["shap"]), reverse=True)

        prob_series = self.test_preds[self.test_preds["Provider"] == provider_id]["y_proba"]
        if len(prob_series) > 0:
            model_prob = float(prob_series.values[0])
        else:
            X_inst = pd.DataFrame([x_row])[self.features]
            model_prob = float(self.model.predict_proba(X_inst)[0][1])

        return {
            "provider": provider_id,
            "base_value": 0.0,
            "shap_values": {f: float(shap_row[i]) for i, f in enumerate(self.features)},
            "feature_values": {f: float(x_row[f]) for f in self.features},
            "top_drivers": drivers[:10],
            "model_probability": round(model_prob, 4),
        }

    def metrics_summary(self) -> dict:
        return {
            "pr_auc": round(float(self.metrics["PR-AUC"]), 4),
            "roc_auc": round(float(self.metrics["ROC-AUC"]), 4),
            "precision": round(float(self.metrics["Precision"]), 4),
            "recall": round(float(self.metrics["Recall"]), 4),
            "f1": round(float(self.metrics["F1"]), 4),
            "threshold": round(float(self.threshold), 4),
            "confusion_matrix": self.metrics.get("Confusion_Matrix"),
            "threshold_comparison": self.model_data.get("threshold_comparison", []),
        }

    def get_demo_provider(self) -> dict:
        return self.demo_provider


service = ModelService()
