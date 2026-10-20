"""
Utility script to find the optimal demo provider for live presentation.
Criteria:
  - Highest fraud probability in the test set
  - At least 3 SHAP feature impacts with |SHAP| > 0.1 for high interpretability
Exports:
  - models/demo_provider.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import shap


def find_demo_provider(
    model_path: str = "models/model.pkl",
    features_path: str = "models/features.parquet",
    test_preds_path: str = "models/test_preds.parquet",
    output_path: str = "models/demo_provider.json",
) -> dict:
    """
    Identify and persist an interpretable, high-risk demo provider.

    Parameters
    ----------
    model_path : str
        Path to saved model artifact.
    features_path : str
        Path to features dataframe.
    test_preds_path : str
        Path to test predictions dataframe.
    output_path : str
        Path to save demo provider JSON.

    Returns
    -------
    dict
        Metadata for selected demo provider.
    """
    model_file = Path(model_path)
    feat_file = Path(features_path)
    preds_file = Path(test_preds_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(model_file)
    model = artifact["model"]
    feature_cols = artifact["features"]

    features_df = pd.read_parquet(feat_file)
    test_preds_df = pd.read_parquet(preds_file)

    # Sort test providers by fraud probability descending
    sorted_test = test_preds_df.sort_values(by="y_proba", ascending=False)

    explainer = shap.TreeExplainer(model)

    chosen_provider = None
    chosen_prob = None
    chosen_actual = None
    chosen_shap = None
    chosen_row = None

    for _, row in sorted_test.iterrows():
        provider_id = row["Provider"]
        prob = row["y_proba"]
        actual = row["y_true"]

        feat_row = features_df.loc[provider_id, feature_cols].values.reshape(1, -1)
        raw_shap = explainer.shap_values(feat_row)

        if isinstance(raw_shap, list):
            shap_vec = raw_shap[1][0]
        elif hasattr(raw_shap, "values"):
            shap_vec = raw_shap.values[0]
        else:
            shap_vec = np.array(raw_shap)[0]

        # Check for at least 3 features with |shap| > 0.1
        significant_feats = np.sum(np.abs(shap_vec) > 0.1)
        if significant_feats >= 3:
            chosen_provider = provider_id
            chosen_prob = float(prob)
            chosen_actual = int(actual)
            chosen_shap = shap_vec
            chosen_row = features_df.loc[provider_id, feature_cols]
            break

    # Fallback to absolute highest probability if threshold not met
    if chosen_provider is None:
        top_row = sorted_test.iloc[0]
        chosen_provider = top_row["Provider"]
        chosen_prob = float(top_row["y_proba"])
        chosen_actual = int(top_row["y_true"])
        feat_row = features_df.loc[chosen_provider, feature_cols].values.reshape(1, -1)
        raw_shap = explainer.shap_values(feat_row)
        chosen_shap = np.array(raw_shap)[0]
        chosen_row = features_df.loc[chosen_provider, feature_cols]

    # Rank top 5 SHAP drivers
    top5_indices = np.argsort(np.abs(chosen_shap))[::-1][:5]
    top_drivers = []
    for idx in top5_indices:
        feat_name = feature_cols[idx]
        top_drivers.append({
            "feature": feat_name,
            "value": float(chosen_row[feat_name]),
            "shap": round(float(chosen_shap[idx]), 4),
        })

    demo_info = {
        "provider": chosen_provider,
        "prob": round(chosen_prob, 4),
        "actual_label": chosen_actual,
        "top_drivers": top_drivers,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(demo_info, f, indent=2)

    print("=" * 60)
    print("DEMO PROVIDER SELECTED")
    print("=" * 60)
    print(f"Provider ID  : {chosen_provider}")
    print(f"Fraud Prob   : {chosen_prob:.4f}")
    print(f"Actual Label : {'Fraud (1)' if chosen_actual == 1 else 'Non-Fraud (0)'}")
    print("\nTop 5 SHAP Drivers:")
    for d in top_drivers:
        sign = "+" if d["shap"] > 0 else ""
        print(f"  - {d['feature']:<25} = {d['value']:>10.2f} ({sign}{d['shap']:.4f} SHAP)")
    print("=" * 60)
    print(f"[SUCCESS] Saved to {out_file}\n")

    return demo_info


if __name__ == "__main__":
    find_demo_provider()
