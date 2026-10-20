"""
Feature engineering module for Health Insurance Fraud Detector.
Aggregates claim-level records into provider-level features,
computes fraud-specific behavioral ratios and diversity metrics,
and merges with provider fraud labels.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


def compute_provider_features(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate claim records to provider-level features without requiring fraud labels.

    Parameters
    ----------
    claims : pd.DataFrame
        Enriched claims records (output of data_prep or batch inference).

    Returns
    -------
    pd.DataFrame
        Provider-level feature matrix indexed by 'Provider'.
    """
    g = claims.groupby("Provider")

    # Safe accessors for columns that might be absent in small custom slices
    def col_mean(col):
        return g[col].mean() if col in claims.columns else 0.0

    def col_sum(col):
        return g[col].sum() if col in claims.columns else 0.0

    def col_max(col):
        return g[col].max() if col in claims.columns else 0.0

    def col_std(col):
        return g[col].std().fillna(0) if col in claims.columns else 0.0

    feats = pd.DataFrame({
        "claim_count": g.size(),
        "total_reimb": col_sum("InscClaimAmtReimbursed"),
        "avg_reimb": col_mean("InscClaimAmtReimbursed"),
        "max_reimb": col_max("InscClaimAmtReimbursed"),
        "std_reimb": col_std("InscClaimAmtReimbursed"),
        "unique_bene": g["BeneID"].nunique() if "BeneID" in claims.columns else 1,
        "unique_attending": g["AttendingPhysician"].nunique() if "AttendingPhysician" in claims.columns else 1,
        "inpatient_ratio": col_mean("IsInpatient"),
        "avg_deductible": col_mean("DeductibleAmtPaid"),
        "avg_claim_duration": col_mean("ClaimDuration"),
        "max_claim_duration": col_max("ClaimDuration"),
        "avg_stay_days": col_mean("StayDays"),
        "avg_age": col_mean("Age"),
        "std_age": col_std("Age"),
        "avg_chronic": col_mean("ChronicCount"),
        "renal_ratio": col_mean("Renal"),
        "dead_bene_ratio": col_mean("IsDead"),
        "missing_attending_ratio": col_mean("MissingAttending"),
        "missing_operating_ratio": col_mean("MissingOperating"),
        "avg_diag_per_claim": col_mean("DiagCount"),
        "max_diag_per_claim": col_max("DiagCount"),
        "avg_proc_per_claim": col_mean("ProcCount"),
        "unique_states": g["State"].nunique() if "State" in claims.columns else 0,
    })

    # Guard against div-by-zero
    eps = 1e-9

    feats["claims_per_bene"] = feats["claim_count"] / (feats["unique_bene"] + eps)
    feats["reimb_per_bene"] = feats["total_reimb"] / (feats["unique_bene"] + eps)
    feats["claims_per_physician"] = feats["claim_count"] / (feats["unique_attending"] + eps)
    feats["physician_per_bene_ratio"] = feats["unique_attending"] / (feats["unique_bene"] + eps)

    # Unique diagnosis and procedure codes per provider
    diag_cols = [c for c in claims.columns if c.startswith("ClmDiagnosisCode_")]
    proc_cols = [c for c in claims.columns if c.startswith("ClmProcedureCode_")]

    if diag_cols:
        diag_melt = claims[["Provider"] + diag_cols].melt("Provider").dropna()
        unique_diag = diag_melt.groupby("Provider")["value"].nunique().reindex(feats.index, fill_value=0)
    else:
        unique_diag = pd.Series(0, index=feats.index)

    if proc_cols:
        proc_melt = claims[["Provider"] + proc_cols].melt("Provider").dropna()
        unique_proc = proc_melt.groupby("Provider")["value"].nunique().reindex(feats.index, fill_value=0)
    else:
        unique_proc = pd.Series(0, index=feats.index)

    feats["diag_diversity"] = unique_diag / (feats["claim_count"] + eps)
    feats["proc_diversity"] = unique_proc / (feats["claim_count"] + eps)
    feats["diagnosis_per_bene"] = unique_diag / (feats["unique_bene"] + eps)

    # Clean missing / infinities
    feats = feats.replace([np.inf, -np.inf], 0).fillna(0)
    return feats


def build_provider_features(
    claims: pd.DataFrame = None,
    labels: pd.DataFrame = None,
    data_dir: str = "data",
    output_dir: str = "models",
) -> pd.DataFrame:
    """
    Aggregate claim records to provider-level features and attach labels.

    Parameters
    ----------
    claims : pd.DataFrame, optional
        Preprocessed claims DataFrame. If None, loaded from claims_processed.parquet.
    labels : pd.DataFrame, optional
        Provider fraud labels DataFrame. If None, loaded from Train.csv.
    data_dir : str
        Directory where claims and labels reside.
    output_dir : str
        Directory to save models/features.parquet.

    Returns
    -------
    pd.DataFrame
        Provider-level feature matrix with target 'y' and Provider index.
    """
    base_data = Path(data_dir)
    base_output = Path(output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    if claims is None:
        parquet_path = base_data / "claims_processed.parquet"
        if not parquet_path.exists():
            print(f"[ERROR] Processed claims file not found: {parquet_path}. Run src/data_prep.py first.", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Reading processed claims from {parquet_path}...")
        claims = pd.read_parquet(parquet_path)

    if labels is None:
        labels_path = base_data / "Train.csv"
        if not labels_path.exists():
            print(f"[ERROR] Provider labels file not found: {labels_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Reading provider labels from {labels_path}...")
        labels = pd.read_csv(labels_path)

    print("[INFO] Aggregating claims to provider level...")
    feats = compute_provider_features(claims)

    # Merge labels (only labeled providers are kept for training/validation)
    labels = labels.copy()
    labels = labels.drop_duplicates(subset=["Provider"]).set_index("Provider")
    labels["y"] = (labels["PotentialFraud"].astype(str).str.strip().str.upper() == "YES").astype(int)

    # Join labels
    feats = feats.join(labels[["y"]], how="inner")

    output_path = base_output / "features.parquet"
    feats.to_parquet(output_path, index=True)

    fraud_count = feats["y"].sum()
    total_count = len(feats)
    fraud_pct = (fraud_count / total_count) * 100 if total_count > 0 else 0

    print(f"\n[SUCCESS] Features saved to {output_path}")
    print(f"[INFO] Features matrix shape: {feats.shape}")
    print(f"[INFO] Class balance: Total={total_count}, Fraud={fraud_count} ({fraud_pct:.2f}%), Non-Fraud={total_count - fraud_count}")
    print("\n--- Features List ---")
    feature_cols = [c for c in feats.columns if c != "y"]
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")

    return feats


if __name__ == "__main__":
    build_provider_features()
