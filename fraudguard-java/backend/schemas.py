from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    provider: str
    fraud_probability: float
    risk_tier: str
    threshold: float


class ProviderPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    provider: str = Field(..., alias="Provider")
    y_true: int
    y_proba: float
    y_pred: int
    risk_tier: str


class ShapDriver(BaseModel):
    feature: str
    value: float
    shap: float


class ShapExplanation(BaseModel):
    provider: str
    base_value: float
    shap_values: Dict[str, float]
    feature_values: Dict[str, float]
    top_drivers: List[ShapDriver]
    model_probability: float


class ModelMetrics(BaseModel):
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    confusion_matrix: Optional[List[List[int]]] = None
    threshold_comparison: Optional[Any] = None
