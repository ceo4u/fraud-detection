"""
Model explainability module using SHAP for Health Insurance Fraud Detector.
Generates SHAP summary plot, top 10 feature importance plot,
exports precomputed SHAP values and sample feature matrix for rapid Streamlit app explanation.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import train_test_split


def generate_explanations(
    model_path: str = "models/model.pkl",
    features_path: str = "models/features.parquet",
    reports_dir: str = "reports",
    models_dir: str = "models",
    sample_size: int = 500,
    random_state: int = 42,
) -> None:
    """
    Generate and save SHAP explanations for the best trained model.

    Parameters
    ----------
    model_path : str
        Path to the saved model dictionary.
    features_path : str
        Path to features.parquet.
    reports_dir : str
        Directory to save SHAP visualizations.
    models_dir : str
        Directory to save SHAP numpy array and sample parquet.
    sample_size : int
        Number of test instances to compute SHAP values on (for speed).
    random_state : int
        Random seed for sampling.
    """
    rep_dir = Path(reports_dir)
    mod_dir = Path(models_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[ERROR] Model artifact {model_file} not found. Run src/train.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading model artifact from {model_file}...")
    saved_payload = joblib.load(model_file)
    model = saved_payload["model"]
    feature_names = saved_payload["features"]
    model_name = saved_payload.get("model_name", "Model")

    # Load features to extract test split matching train.py
    features_file = Path(features_path)
    if not features_file.exists():
        print(f"[ERROR] Features file {features_file} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(features_file)
    X = df[feature_names]
    y = df["y"].astype(int)

    # Recreate the exact stratified test split
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Sample for fast, responsive SHAP computation
    if len(X_test) > sample_size:
        X_sample = X_test.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X_test.copy()

    print(f"[INFO] Computing SHAP values on {len(X_sample)} sample test providers...")

    if "XGB" in model_name or hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_sample)
    else:
        # LogisticRegression Pipeline
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        X_sample_scaled = scaler.transform(X_sample)
        explainer = shap.LinearExplainer(clf, X_sample_scaled)
        raw_shap = explainer.shap_values(X_sample_scaled)

    # Handle shape differences across SHAP versions (binary classification)
    if isinstance(raw_shap, list):
        shap_vals = raw_shap[1]  # positive fraud class
    elif hasattr(raw_shap, "values"):
        shap_vals = raw_shap.values
    else:
        shap_vals = np.array(raw_shap)

    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    # Save SHAP artifacts for Streamlit app
    shap_vals_path = mod_dir / "shap_values.npy"
    shap_sample_path = mod_dir / "shap_X_sample.parquet"

    np.save(shap_vals_path, shap_vals)
    X_sample.to_parquet(shap_sample_path, index=True)
    print(f"[SUCCESS] Saved SHAP values to {shap_vals_path}")
    print(f"[SUCCESS] Saved sample features to {shap_sample_path}")

    # 1. SHAP Summary Plot
    summary_plot_path = rep_dir / "shap_summary.png"
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved SHAP summary plot to {summary_plot_path}")

    # 2. Top 10 Feature Importance Bar Plot
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
    top10_names = [feature_names[i] for i in top10_idx]
    top10_scores = mean_abs_shap[top10_idx]

    top10_plot_path = rep_dir / "top10_importance.png"
    plt.figure(figsize=(9, 5))
    y_pos = np.arange(len(top10_names))
    plt.barh(y_pos, top10_scores[::-1], color="#1f77b4", edgecolor="none", alpha=0.85)
    plt.yticks(y_pos, top10_names[::-1], fontsize=10)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Fraud Prediction)", fontsize=11)
    plt.title("Top 10 Fraud Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(top10_plot_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved top 10 importance plot to {top10_plot_path}")

    # Print top 10 features
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES BY SHAP IMPORTANCE")
    print("=" * 60)
    for rank, (name, score) in enumerate(zip(top10_names, top10_scores), 1):
        print(f"  {rank:2d}. {name:<30} {score:.5f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    generate_explanations()
