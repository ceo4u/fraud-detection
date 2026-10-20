"""
Data preparation module for Health Insurance Fraud Detector.
Reads inpatient, outpatient, beneficiary, and provider label data,
performs date parsing and joins, computes claim-level derived fields,
and exports processed claims to parquet.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


def prepare_claims(
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Tag claim types, combine inpatient and outpatient claims, and parse claim dates.

    Parameters
    ----------
    inpatient_df : pd.DataFrame
        Inpatient claims records.
    outpatient_df : pd.DataFrame, optional
        Outpatient claims records.

    Returns
    -------
    pd.DataFrame
        Combined claims DataFrame with standardized dates and claim types.
    """
    ip = inpatient_df.copy()
    ip["ClaimType"] = "Inpatient"

    if outpatient_df is not None and not outpatient_df.empty:
        op = outpatient_df.copy()
        op["ClaimType"] = "Outpatient"
        claims = pd.concat([ip, op], ignore_index=True, sort=False)
    else:
        claims = ip

    claim_date_cols = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]
    for col in claim_date_cols:
        if col in claims.columns:
            claims[col] = pd.to_datetime(claims[col], errors="coerce")

    return claims


def merge_beneficiary(
    claims: pd.DataFrame,
    bene: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse beneficiary dates, join beneficiary demographics to claims,
    and compute claim-level derived features.

    Parameters
    ----------
    claims : pd.DataFrame
        Combined claims DataFrame from prepare_claims().
    bene : pd.DataFrame
        Beneficiary records.

    Returns
    -------
    pd.DataFrame
        Enriched claims DataFrame.
    """
    claims = claims.copy()
    bene = bene.copy()

    bene_date_cols = ["DOB", "DOD"]
    for col in bene_date_cols:
        if col in bene.columns:
            bene[col] = pd.to_datetime(bene[col], errors="coerce")

    # Merge beneficiary demographics onto claims
    claims = claims.merge(bene, on="BeneID", how="left")

    # Compute derived columns
    claims["Age"] = (claims["ClaimStartDt"] - claims["DOB"]).dt.days / 365.25
    claims["ClaimDuration"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days
    if "DischargeDt" in claims.columns and "AdmissionDt" in claims.columns:
        claims["StayDays"] = (claims["DischargeDt"] - claims["AdmissionDt"]).dt.days.fillna(0)
    else:
        claims["StayDays"] = 0

    claims["IsInpatient"] = (claims["ClaimType"] == "Inpatient").astype(int)

    # Chronic conditions (1 = Yes, 2 = No in CMS dataset)
    chronic_cols = [c for c in claims.columns if c.startswith("ChronicCond_")]
    for col in chronic_cols:
        claims[col] = (claims[col] == 1).astype(int)
    claims["ChronicCount"] = claims[chronic_cols].sum(axis=1) if chronic_cols else 0

    # Renal disease ('Y' or '0')
    if "RenalDiseaseIndicator" in claims.columns:
        claims["Renal"] = (claims["RenalDiseaseIndicator"].astype(str).str.strip().str.upper() == "Y").astype(int)
    else:
        claims["Renal"] = 0

    # Beneficiary deceased flag
    if "DOD" in claims.columns:
        claims["IsDead"] = claims["DOD"].notna().astype(int)
    else:
        claims["IsDead"] = 0

    # Missing physician flags
    if "AttendingPhysician" in claims.columns:
        claims["MissingAttending"] = claims["AttendingPhysician"].isna().astype(int)
    else:
        claims["MissingAttending"] = 0

    if "OperatingPhysician" in claims.columns:
        claims["MissingOperating"] = claims["OperatingPhysician"].isna().astype(int)
    else:
        claims["MissingOperating"] = 1

    # Diagnosis and procedure code counts
    diag_cols = [c for c in claims.columns if c.startswith("ClmDiagnosisCode_")]
    proc_cols = [c for c in claims.columns if c.startswith("ClmProcedureCode_")]

    claims["DiagCount"] = claims[diag_cols].notna().sum(axis=1) if diag_cols else 0
    claims["ProcCount"] = claims[proc_cols].notna().sum(axis=1) if proc_cols else 0

    return claims


def build_claims(data_dir: str = "data") -> pd.DataFrame:
    """
    Build unified processed claims dataset from raw CSV files.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw CSV files.

    Returns
    -------
    pd.DataFrame
        Merged and enriched claims DataFrame.
    """
    base_path = Path(data_dir)

    beneficiary_path = base_path / "Train_Beneficiarydata.csv"
    inpatient_path = base_path / "Train_Inpatientdata.csv"
    outpatient_path = base_path / "Train_Outpatientdata.csv"
    labels_path = base_path / "Train.csv"

    required_files = [
        beneficiary_path,
        inpatient_path,
        outpatient_path,
        labels_path,
    ]

    missing_files = [str(f) for f in required_files if not f.exists()]
    if missing_files:
        print(f"[ERROR] Missing required dataset files:\n" + "\n".join(f"  - {f}" for f in missing_files), file=sys.stderr)
        sys.exit(1)

    print("[INFO] Loading raw datasets...")
    bene = pd.read_csv(beneficiary_path)
    ip = pd.read_csv(inpatient_path)
    op = pd.read_csv(outpatient_path)

    print(f"[INFO] Beneficiaries: {bene.shape}, Inpatient claims: {ip.shape}, Outpatient claims: {op.shape}")

    claims = prepare_claims(ip, op)
    print(f"[INFO] Combined raw claims: {claims.shape}")

    claims = merge_beneficiary(claims, bene)
    print(f"[INFO] Claims merged with Beneficiary data: {claims.shape}")

    # Export to parquet
    output_path = base_path / "claims_processed.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    claims.to_parquet(output_path, index=False)

    print(f"\n[SUCCESS] Saved processed claims to {output_path}")
    print(f"[INFO] Processed claims shape: {claims.shape}")
    print("\n--- Missing Value Summary (Top 15) ---")
    print(claims.isnull().sum().sort_values(ascending=False).head(15))
    print("\n--- Head (5 Rows) ---")
    print(claims[["BeneID", "ClaimID", "Provider", "ClaimType", "Age", "ClaimDuration", "StayDays", "ChronicCount", "DiagCount"]].head())

    return claims


if __name__ == "__main__":
    build_claims()
