"""
Batch inference module for Health Insurance Fraud Detector.
Enables scoring of raw, multi-table claims CSVs without manual feature computation.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, List
import pandas as pd
import numpy as np
import joblib

from src.data_prep import prepare_claims, merge_beneficiary
from src.features import compute_provider_features

REQUIRED_INPATIENT_COLS = [
    "BeneID",
    "ClaimID",
    "ClaimStartDt",
    "ClaimEndDt",
    "Provider",
    "InscClaimAmtReimbursed",
]

REQUIRED_BENEFICIARY_COLS = [
    "BeneID",
    "DOB",
    "Gender",
    "Race",
    "State",
]


def validate_upload(
    df: pd.DataFrame,
    expected_cols: List[str],
    name: str = "Uploaded CSV",
) -> Tuple[bool, str]:
    """
    Validate that an uploaded DataFrame contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded DataFrame to validate.
    expected_cols : list of str
        List of required column names.
    name : str
        Human-readable name of the dataset for error messages.

    Returns
    -------
    tuple of (bool, str)
        (True, "") if valid, or (False, error_message) if invalid.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, f"{name}: No DataFrame provided or empty file."

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        msg = f"{name} is missing {len(missing)} required column(s): {', '.join(missing[:8])}"
        if len(missing) > 8:
            msg += f" ...and {len(missing) - 8} more."
        return False, msg

    if len(df) == 0:
        return False, f"{name} contains 0 rows."

    return True, ""


def score_raw_claims(
    inpatient_df: pd.DataFrame,
    outpatient_df: Optional[pd.DataFrame] = None,
    beneficiary_df: Optional[pd.DataFrame] = None,
    model_data: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Score raw claims tables end-to-end to generate provider-level fraud risk predictions.

    Parameters
    ----------
    inpatient_df : pd.DataFrame
        Raw inpatient claims DataFrame (must contain BeneID, ClaimID, Provider, etc.).
    outpatient_df : pd.DataFrame, optional
        Raw outpatient claims DataFrame.
    beneficiary_df : pd.DataFrame
        Raw beneficiary demographics DataFrame (must contain BeneID, DOB, etc.).
    model_data : dict, optional
        Pre-loaded model artifact dictionary. If None, loads from models/model.pkl.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['Provider', 'fraud_probability', 'risk_tier', 'top_drivers']
        sorted by fraud_probability descending.
    """
    # 1. Validate required datasets
    ok_ip, err_ip = validate_upload(inpatient_df, REQUIRED_INPATIENT_COLS, "Inpatient claims")
    if not ok_ip:
        raise ValueError(err_ip)

    if beneficiary_df is None:
        raise ValueError("Beneficiary dataset is required for scoring.")

    ok_bene, err_bene = validate_upload(beneficiary_df, REQUIRED_BENEFICIARY_COLS, "Beneficiary data")
    if not ok_bene:
        raise ValueError(err_bene)

    # 2. Prepare and combine claims
    claims = prepare_claims(inpatient_df, outpatient_df)

    # 3. Merge beneficiary demographics & compute claim-level fields
    claims_enriched = merge_beneficiary(claims, beneficiary_df)

    # 4. Aggregate to provider-level features
    provider_feats = compute_provider_features(claims_enriched)

    if provider_feats.empty:
        return pd.DataFrame(columns=["Provider", "fraud_probability", "risk_tier", "top_drivers"])

    # 5. Load model artifact if not supplied
    if model_data is None:
        model_path = Path("models/model.pkl")
        if not model_path.exists():
            # Check relative to repo
            alt_path = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
            if alt_path.exists():
                model_path = alt_path
            else:
                raise FileNotFoundError("models/model.pkl not found. Please train model first.")
        model_data = joblib.load(model_path)

    model = model_data["model"]
    feature_cols = model_data["features"]
    threshold = float(model_data.get("threshold", 0.28))

    # 6. Align feature matrix
    X = provider_feats.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)

    # 7. Model Inference
    probs = model.predict_proba(X)[:, 1]

    # 8. Determine Risk Tiers
    def get_tier(p: float) -> str:
        if p >= threshold:
            return "High"
        elif p >= 0.5 * threshold:
            return "Medium"
        return "Low"

    tiers = [get_tier(p) for p in probs]

    # 9. Compute Top Behavioral Drivers
    # Weight provider feature values by global model feature importances
    top_drivers_list = []
    feat_imp = getattr(model, "feature_importances_", None)
    if feat_imp is not None:
        imp_series = pd.Series(feat_imp, index=feature_cols)
        # Compute feature impact score for non-zero features
        for _, row in X.iterrows():
            pos_signals = row[row > 0]
            if len(pos_signals) > 0:
                weighted_impact = pos_signals * imp_series.reindex(pos_signals.index, fill_value=0.0)
                top_3 = weighted_impact.sort_values(ascending=False).head(3).index.tolist()
                top_drivers_list.append(", ".join(top_3))
            else:
                top_drivers_list.append("None identified")
    else:
        top_drivers_list = ["N/A"] * len(X)

    # 10. Assemble output DataFrame
    results = pd.DataFrame({
        "Provider": provider_feats.index,
        "fraud_probability": np.round(probs, 4),
        "risk_tier": tiers,
        "top_drivers": top_drivers_list,
    }).sort_values(by="fraud_probability", ascending=False).reset_index(drop=True)

    return results
