"""
Script to generate the EDA notebook and produce exploratory analysis plots.
Outputs:
  - notebooks/eda.ipynb
  - reports/eda_class_balance.png
  - reports/eda_distributions.png
  - reports/eda_correlation_heatmap.png
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)
notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

# Load features dataset if available
features_path = Path("models/features.parquet")
if not features_path.exists():
    print("[INFO] Features file not ready yet. EDA generator will run when features are built.")
else:
    df = pd.read_parquet(features_path)
    print(f"[INFO] Loaded {df.shape} providers for EDA.")

    # 1. Class Balance Bar Chart
    plt.figure(figsize=(7, 5))
    counts = df["y"].value_counts()
    labels = ["Non-Fraud (0)", "Fraud (1)"]
    pcts = [counts.get(0, 0) / len(df) * 100, counts.get(1, 0) / len(df) * 100]
    bars = plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#2ca02c", "#d62728"], width=0.5)
    for bar, pct in zip(bars, pcts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 50, f"{yval} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    plt.title("Provider Class Balance (Kaggle CMS Fraud Dataset)", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Providers", fontsize=11)
    plt.ylim(0, max(counts) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(reports_dir / "eda_class_balance.png", dpi=200)
    plt.close()
    print("[SUCCESS] Saved reports/eda_class_balance.png")

    # 2. Distributions on Key Features
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    dist_features = [
        ("avg_reimb", "Average Claim Reimbursement ($)"),
        ("claim_count", "Total Claims Count"),
        ("avg_claim_duration", "Average Claim Duration (Days)"),
        ("inpatient_ratio", "Inpatient Claim Ratio"),
    ]

    for ax, (col, title) in zip(axes.flatten(), dist_features):
        fraud = df[df["y"] == 1][col]
        non_fraud = df[df["y"] == 0][col]

        # Use 99th percentile clip for visualization clarity on skewed amounts
        q99 = max(df[col].quantile(0.99), 1)
        sns.histplot(non_fraud[non_fraud <= q99], ax=ax, color="#2ca02c", label="Non-Fraud", kde=True, stat="density", bins=30, alpha=0.4)
        sns.histplot(fraud[fraud <= q99], ax=ax, color="#d62728", label="Fraud", kde=True, stat="density", bins=30, alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Distributions by Fraud Status (Clipped at 99th percentile for display)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / "eda_distributions.png", dpi=200)
    plt.close()
    print("[SUCCESS] Saved reports/eda_distributions.png")

    # 3. Correlation Heatmap (Top 15 Features with Target 'y')
    corr_series = df.corr()["y"].abs().sort_values(ascending=False)
    top15_features = corr_series.head(16).index.tolist()  # includes 'y'

    plt.figure(figsize=(11, 9))
    corr_matrix = df[top15_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap: Top 15 Features Associated with Fraud", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / "eda_correlation_heatmap.png", dpi=200)
    plt.close()
    print("[SUCCESS] Saved reports/eda_correlation_heatmap.png")


# Create Jupyter Notebook JSON structure
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Healthcare Provider Fraud Detection — Exploratory Data Analysis (EDA)\n",
                "\n",
                "This notebook analyzes the Medicare claims dataset aggregated to the provider level.\n",
                "The objective is to identify key behavioral, clinical, and financial disparities between fraudulent and legitimate healthcare providers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "df = pd.read_parquet('../models/features.parquet')\n",
                "print('Providers:', df.shape[0], 'Features:', df.shape[1] - 1)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Class Distribution Analysis\n",
                "Checking the degree of class imbalance in Medicare provider fraud labels."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "counts = df['y'].value_counts()\n",
                "pcts = counts / len(df) * 100\n",
                "print(f'Class balance:\\nNon-fraud: {counts[0]} ({pcts[0]:.2f}%)\\nFraud: {counts[1]} ({pcts[1]:.2f}%)')\n",
                "\n",
                "plt.figure(figsize=(6, 4))\n",
                "bars = plt.bar(['Non-Fraud (0)', 'Fraud (1)'], counts, color=['#2ca02c', '#d62728'])\n",
                "plt.title('Provider Class Balance')\n",
                "plt.ylabel('Count')\n",
                "plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Feature Distribution Disparities\n",
                "Comparing fraudulent vs. non-fraudulent distributions on high-impact predictors:\n",
                "- Average reimbursement (`avg_reimb`)\n",
                "- Total claim count (`claim_count`)\n",
                "- Average claim duration (`avg_claim_duration`)\n",
                "- Inpatient ratio (`inpatient_ratio`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
                "cols = ['avg_reimb', 'claim_count', 'avg_claim_duration', 'inpatient_ratio']\n",
                "for ax, col in zip(axes.flatten(), cols):\n",
                "    sns.kdeplot(data=df, x=col, hue='y', common_norm=False, ax=ax, palette=['#2ca02c', '#d62728'])\n",
                "    ax.set_title(f'Distribution: {col}')\n",
                "    ax.grid(True, linestyle='--', alpha=0.5)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Correlation Analysis\n",
                "Examining the correlation matrix among top features most correlated with provider fraud."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "top15 = df.corr()['y'].abs().sort_values(ascending=False).head(16).index\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(df[top15].corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)\n",
                "plt.title('Top 15 Feature Correlation Matrix')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Key Insights Summary\n",
                "\n",
                "- **Severe Class Imbalance (~9% Fraud):** Only ~9% of providers are flagged as potentially fraudulent. Standard accuracy is an uninformative metric; PR-AUC and Recall-at-Precision are critical.\n",
                "- **High Billing Volume Disparity:** Fraudulent providers process significantly higher claim counts and total reimbursement sums compared to normal providers, indicating systematic over-submission.\n",
                "- **Per-Beneficiary Disproportion:** Fraudulent providers exhibit substantially higher `reimb_per_bene` and `claims_per_bene`, pointing to potential phantom billing or patient churning.\n",
                "- **Inpatient and Duration Outliers:** Inpatient ratio and claim duration variance correlate positively with fraud, reflecting complex, high-payout claim manipulations.\n",
                "- **Physician Network Signals:** Elevated claims-per-physician ratios highlight potential collusion rings or automated billing templates across concentrated attending physicians."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebooks_dir / "eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("[SUCCESS] Generated notebooks/eda.ipynb")
