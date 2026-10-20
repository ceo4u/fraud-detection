"""
Model training and evaluation module for Health Insurance Fraud Detector.
Trains Logistic Regression baseline and XGBoost classifier with stratified K-fold cross-validation.
Tunes classification thresholds on out-of-fold predictions to prevent test leakage:
  - Recall-first threshold (precision >= 0.30)
  - Balanced threshold (precision >= 0.50, selected production threshold)
  - F1-optimal threshold (maximum F1)
Evaluates on held-out test split, exports the best model and prediction artifacts.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier


def train_models(
    features_path: str = "models/features.parquet",
    output_dir: str = "models",
    random_state: int = 42,
) -> dict:
    """
    Train and evaluate fraud detection models with 3-tier threshold tuning.

    Parameters
    ----------
    features_path : str
        Path to features.parquet file.
    output_dir : str
        Directory to save model artifacts and predictions.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing trained models, comparison metrics, and best model name.
    """
    features_file = Path(features_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_file.exists():
        print(f"[ERROR] Features file {features_file} not found. Run src/features.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading features from {features_file}...")
    df = pd.read_parquet(features_file)

    if "y" not in df.columns:
        print("[ERROR] Target column 'y' not found in features DataFrame.", file=sys.stderr)
        sys.exit(1)

    X = df.drop(columns=["y"])
    y = df["y"].astype(int)

    print(f"[INFO] Features dataset: X={X.shape}, y={y.shape}. Positive fraud rate: {y.mean():.4f}")

    # Train / test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_state,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}
    fitted_models = {}
    test_probs = {}
    threshold_data = {}

    for name, model in models.items():
        print(f"\n[INFO] Cross-validating {name}...")
        oof_proba = cross_val_predict(
            model, X_train, y_train, cv=skf, method="predict_proba"
        )[:, 1]

        # OOF Threshold Tuning (preventing test leakage)
        precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        best_idx_f1 = int(np.argmax(f1_scores[:-1]))
        thr_f1_optimal = float(thresholds[best_idx_f1])

        # 1. Recall-first threshold: lowest threshold where precision >= 0.30
        valid_30 = np.where(precisions[:-1] >= 0.30)[0]
        thr_recall_first = float(thresholds[valid_30[0]]) if len(valid_30) > 0 else thr_f1_optimal

        # 2. Balanced threshold: lowest threshold where precision >= 0.50 (NEW - Production)
        valid_50 = np.where(precisions[:-1] >= 0.50)[0]
        thr_balanced = float(thresholds[valid_50[0]]) if len(valid_50) > 0 else thr_f1_optimal

        print(f"[{name}] OOF Thresholds -> Recall-first (prec>=0.30): {thr_recall_first:.4f} | Balanced (prec>=0.50): {thr_balanced:.4f} | F1-optimal: {thr_f1_optimal:.4f}")

        # Fit model on entire X_train
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Evaluate on held-out X_test
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_probs[name] = y_test_proba

        # Evaluate all three thresholds on held-out X_test
        thrs = {
            "Recall-first (prec>=0.30)": thr_recall_first,
            "Balanced (prec>=0.50) [PROD]": thr_balanced,
            "F1-optimal": thr_f1_optimal,
        }

        thrs_eval = {}
        for strat_name, thr_val in thrs.items():
            y_pred_strat = (y_test_proba >= thr_val).astype(int)
            thrs_eval[strat_name] = {
                "Threshold": thr_val,
                "Precision": float(precision_score(y_test, y_pred_strat, zero_division=0)),
                "Recall": float(recall_score(y_test, y_pred_strat, zero_division=0)),
                "F1": float(f1_score(y_test, y_pred_strat, zero_division=0)),
            }

        threshold_data[name] = thrs_eval

        # Production metrics based on thr_balanced
        y_test_pred_prod = (y_test_proba >= thr_balanced).astype(int)
        pr_auc = float(average_precision_score(y_test, y_test_proba))
        roc_auc = float(roc_auc_score(y_test, y_test_proba))
        prec = float(precision_score(y_test, y_test_pred_prod, zero_division=0))
        rec = float(recall_score(y_test, y_test_pred_prod, zero_division=0))
        f1 = float(f1_score(y_test, y_test_pred_prod, zero_division=0))
        cm = confusion_matrix(y_test, y_test_pred_prod)

        results[name] = {
            "PR-AUC": pr_auc,
            "ROC-AUC": roc_auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Threshold": thr_balanced,
            "thr_balanced": thr_balanced,
            "thr_recall_first": thr_recall_first,
            "thr_f1_optimal": thr_f1_optimal,
            "F1_Threshold": thr_f1_optimal,
            "Confusion_Matrix": cm.tolist(),
        }

    # Model comparison table at production threshold (thr_balanced)
    comp_df = pd.DataFrame(results).T[["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Threshold", "thr_f1_optimal"]]
    print("\n" + "=" * 65)
    print("MODEL COMPARISON (Test Set @ Balanced Production Threshold)")
    print("=" * 65)
    print(comp_df.to_string())
    print("=" * 65 + "\n")

    # Pick best model by PR-AUC
    best_name = max(results.keys(), key=lambda k: results[k]["PR-AUC"])
    best_model = fitted_models[best_name]
    best_res = results[best_name]
    best_thr = best_res["thr_balanced"]
    best_thr_rf = best_res["thr_recall_first"]
    best_thr_f1 = best_res["thr_f1_optimal"]
    best_proba = test_probs[best_name]
    best_pred = (best_proba >= best_thr).astype(int)

    # Threshold comparison table for the best model
    best_thrs_df = pd.DataFrame(threshold_data[best_name]).T[["Threshold", "Precision", "Recall", "F1"]]
    print("=" * 65)
    print(f"THRESHOLD COMPARISON ON TEST SET ({best_name})")
    print("=" * 65)
    print(best_thrs_df.to_string())
    print("=" * 65 + "\n")

    print(f"[INFO] Selected Best Model: {best_name} (PR-AUC: {best_res['PR-AUC']:.4f})")

    # Save model artifact
    model_payload = {
        "model_name": best_name,
        "model": best_model,
        "threshold": best_thr,
        "thr_balanced": best_thr,
        "thr_recall_first": best_thr_rf,
        "thr_f1_optimal": best_thr_f1,
        "f1_threshold": best_thr_f1,
        "features": list(X.columns),
        "metrics": best_res,
        "threshold_comparison": threshold_data[best_name],
        "all_results": results,
    }
    model_out = out_dir / "model.pkl"
    joblib.dump(model_payload, model_out)
    print(f"[SUCCESS] Model artifact saved to {model_out}")

    # Save test predictions (using production threshold)
    test_preds_df = pd.DataFrame({
        "Provider": X_test.index,
        "y_true": y_test.values,
        "y_proba": best_proba,
        "y_pred": best_pred,
    })
    preds_out = out_dir / "test_preds.parquet"
    test_preds_df.to_parquet(preds_out, index=False)
    print(f"[SUCCESS] Test predictions saved to {preds_out}")

    # Required final print line
    print(
        f"Model PR-AUC: {best_res['PR-AUC']:.2f} | "
        f"Threshold: {best_thr:.2f} | "
        f"Recall: {best_res['Recall']:.2f} | "
        f"Precision: {best_res['Precision']:.2f}"
    )

    return {
        "best_name": best_name,
        "models": fitted_models,
        "results": results,
        "threshold_data": threshold_data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    train_models()
