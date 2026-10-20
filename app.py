"""
Healthcare Provider Fraud Detector - Interactive Streamlit Application.
Provides provider risk scoring, individual SHAP explanations, model evaluation metrics,
and batch raw claims inference without manual feature engineering.
All data and predictions are derived directly from trained models and benchmark CMS data.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.batch_inference import score_raw_claims, validate_upload


@st.cache_resource
def load_model_artifact():
    path = Path("models/model.pkl")
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_features_dataset():
    path = Path("models/features.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_test_predictions():
    path = Path("models/test_preds.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_shap_artifacts():
    sample_path = Path("models/shap_X_sample.parquet")
    vals_path = Path("models/shap_values.npy")
    if not sample_path.exists() or not vals_path.exists():
        return None, None
    sample_df = pd.read_parquet(sample_path)
    vals = np.load(vals_path)
    return sample_df, vals


def load_demo_provider_json():
    demo_file = Path("models/demo_provider.json")
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    st.set_page_config(
        page_title="Healthcare Provider Fraud Detector",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Healthcare Provider Fraud Detector")
    st.markdown("CMS Medicare Claims Fraud Detection & Behavioral Explainability Engine")

    artifact = load_model_artifact()
    if artifact is None:
        st.error("models/model.pkl not found. Run python src/train.py first.")
        st.stop()

    features_df = load_features_dataset()
    if features_df is None:
        st.error("models/features.parquet not found. Run python src/features.py first.")
        st.stop()

    test_preds = load_test_predictions()
    if test_preds is None:
        st.error("models/test_preds.parquet not found. Run python src/train.py first.")
        st.stop()

    shap_sample_df, shap_values_arr = load_shap_artifacts()
    demo_info = load_demo_provider_json()

    model = artifact["model"]
    feature_cols = artifact["features"]
    default_thr = float(artifact.get("threshold", 0.28))

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    data_source = st.sidebar.radio(
        "Data Source",
        options=["Test set (1,082 providers)", "All providers (5,410)", "Upload CSV"],
        index=0,
    )

    risk_threshold = st.sidebar.slider(
        "Risk threshold",
        min_value=0.05,
        max_value=0.95,
        value=default_thr,
        step=0.01,
        help="Providers with fraud probability >= threshold will be flagged as High Risk.",
    )

    st.sidebar.markdown(f"**Production Threshold (Balanced):** `{default_thr:.2f}`")

    # Data sourcing logic
    active_df = None

    if data_source == "Test set (1,082 providers)":
        active_df = features_df.loc[test_preds["Provider"]].copy()
    elif data_source == "All providers (5,410)":
        active_df = features_df.copy()
    else:
        st.sidebar.caption("💡 **Tip:** Raw claims upload karne ke liye upar **📤 Batch Score** tab use karein.")
        uploaded_file = st.sidebar.file_uploader("Upload Provider CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if "Provider" not in uploaded_df.columns:
                    st.sidebar.error("Uploaded CSV must contain a 'Provider' column.")
                elif "Fraud_Probability" in uploaded_df.columns or "fraud_probability" in uploaded_df.columns:
                    # User uploaded a predictions CSV directly! Show it gracefully.
                    prob_col = "Fraud_Probability" if "Fraud_Probability" in uploaded_df.columns else "fraud_probability"
                    uploaded_df["Fraud Probability"] = uploaded_df[prob_col]
                    uploaded_df = uploaded_df.set_index("Provider")
                    st.sidebar.success("✅ Loaded pre-scored predictions CSV!")
                    active_df = uploaded_df
                else:
                    missing_cols = [c for c in feature_cols if c not in uploaded_df.columns]
                    if missing_cols:
                        st.sidebar.error(
                            f"Missing {len(missing_cols)} required feature column(s): {', '.join(missing_cols[:3])}...\n\n"
                            f"👉 **Raw claims CSV upload karni hai?** Upar **'📤 Batch Score'** tab pe click karein!"
                        )
                    else:
                        uploaded_df = uploaded_df.set_index("Provider")
                        nan_row_ratio = uploaded_df[feature_cols].isna().any(axis=1).mean()
                        if nan_row_ratio > 0.20:
                            st.sidebar.error("Uploaded file has too many missing values (>20% rows with NaN) — please check.")
                            active_df = None
                        else:
                            if uploaded_df[feature_cols].isna().any().any():
                                st.sidebar.info("Uploaded data contained missing values; filled with 0.")
                                uploaded_df[feature_cols] = uploaded_df[feature_cols].fillna(0)
                            active_df = uploaded_df
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {e}")
        else:
            st.info("👆 Please upload a CSV file with provider features in the sidebar, or use the **📤 Batch Score** tab for raw claims.")
            active_df = None

    # 5 Application Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Predictions",
        "🧠 Explain provider",
        "📊 Model performance",
        "📤 Batch Score",
        "ℹ️ About",
    ])

    if active_df is not None:
        if "Fraud Probability" in active_df.columns:
            probs = active_df["Fraud Probability"].values
            X_active = None
        else:
            X_active = active_df[feature_cols].fillna(0)
            probs = model.predict_proba(X_active)[:, 1]

        pred_df = pd.DataFrame({
            "Provider": active_df.index,
            "Fraud Probability": probs,
        })

        def assign_tier(p):
            if p >= risk_threshold:
                return "High"
            elif p >= 0.5 * risk_threshold:
                return "Medium"
            else:
                return "Low"

        pred_df["Risk Tier"] = pred_df["Fraud Probability"].apply(assign_tier)
        pred_df = pred_df.sort_values(by="Fraud Probability", ascending=False).reset_index(drop=True)

        # Tab 1: Predictions
        with tab1:
            st.subheader("Provider Risk Scoring")

            col1, col2, col3 = st.columns(3)
            total_scanned = len(pred_df)
            high_risk_count = int((pred_df["Risk Tier"] == "High").sum())
            avg_risk_val = float(probs.mean())

            col1.metric("Providers Scanned", f"{total_scanned:,}")
            col2.metric("Flagged High Risk", f"{high_risk_count:,}", f"{(high_risk_count/total_scanned)*100:.1f}%")
            col3.metric("Average Risk Score", f"{avg_risk_val:.2%}")

            st.markdown(f"**Top 50 Providers by Risk Score** (Threshold = `{risk_threshold:.2f}`)")

            display_df = pred_df.head(50).copy()
            display_df.insert(0, "Rank", range(1, len(display_df) + 1))
            display_df["Fraud Probability"] = display_df["Fraud Probability"].apply(lambda v: f"{v:.4f}")

            def color_tier(val):
                if val == "High":
                    return "background-color: #ffcccc; color: #900C3F; font-weight: bold;"
                elif val == "Medium":
                    return "background-color: #fff2cc; color: #B7950B; font-weight: bold;"
                return "background-color: #d4edda; color: #155724;"

            styled_table = display_df.style.map(color_tier, subset=["Risk Tier"])
            st.dataframe(styled_table, use_container_width=True)

            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv_bytes,
                file_name="provider_fraud_predictions.csv",
                mime="text/csv",
            )

        # Tab 2: Explain provider
        with tab2:
            st.subheader("Provider-Level Behavioral Explainability")
            provider_options = list(pred_df["Provider"])

            if demo_info is not None:
                demo_pid = demo_info.get("provider", "")
                c_btn, c_txt = st.columns([1, 4])
                with c_btn:
                    if st.button("⭐ Load Demo Provider", help=f"Pre-selects validated case {demo_pid}"):
                        st.session_state["selected_provider"] = demo_pid
                with c_txt:
                    st.caption(f"Curated High-Risk Case: **{demo_pid}** (Fraud Prob: `{demo_info.get('prob', 0):.4f}`, True Label: Fraud)")

            default_idx = 0
            if "selected_provider" in st.session_state and st.session_state["selected_provider"] in provider_options:
                default_idx = provider_options.index(st.session_state["selected_provider"])

            selected_provider = st.selectbox(
                "Select Provider to Inspect:",
                options=provider_options,
                index=default_idx,
                key="prov_dropdown",
            )
            st.session_state["selected_provider"] = selected_provider

            selected_prob = float(pred_df.loc[pred_df["Provider"] == selected_provider, "Fraud Probability"].iloc[0])
            selected_tier = pred_df.loc[pred_df["Provider"] == selected_provider, "Risk Tier"].iloc[0]

            st.write(f"**Fraud Probability:** `{selected_prob:.4f}` | **Risk Tier:** `{selected_tier}`")

            if X_active is not None and selected_provider in X_active.index:
                provider_row = X_active.loc[selected_provider]
                provider_shap = None

            if shap_sample_df is not None and selected_provider in shap_sample_df.index:
                idx = shap_sample_df.index.get_loc(selected_provider)
                provider_shap = shap_values_arr[idx]
            else:
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    res = explainer.shap_values(provider_row.values.reshape(1, -1))
                    if isinstance(res, list):
                        provider_shap = res[1][0]
                    elif hasattr(res, "values"):
                        provider_shap = res.values[0]
                    else:
                        provider_shap = np.array(res)[0]
                except Exception as ex:
                    st.error(f"Live SHAP calculation error: {ex}")

            if provider_shap is not None:
                import shap
                try:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    explanation = shap.Explanation(
                        values=provider_shap,
                        base_values=0.0,
                        data=provider_row.values,
                        feature_names=feature_cols,
                    )
                    shap.plots.waterfall(explanation, max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    top_idx = np.argsort(np.abs(provider_shap))[::-1][:10]
                    ax.barh([feature_cols[i] for i in top_idx][::-1], provider_shap[top_idx][::-1])
                    ax.set_title(f"SHAP Impacts for {selected_provider}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                sorted_indices = np.argsort(provider_shap)
                top_neg_idx = sorted_indices[:5]
                top_pos_idx = sorted_indices[::-1][:5]

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 🔺 Top Factors Increasing Fraud Risk:")
                    for i in top_pos_idx:
                        if provider_shap[i] > 0:
                            st.markdown(f"- **{feature_cols[i]}**: `{provider_row[feature_cols[i]]:.2f}` *(+{provider_shap[i]:.4f} SHAP)*")

                with c2:
                    st.markdown("##### 🔻 Top Factors Decreasing Fraud Risk:")
                    for i in top_neg_idx:
                        if provider_shap[i] < 0:
                            st.markdown(f"- **{feature_cols[i]}**: `{provider_row[feature_cols[i]]:.2f}` *({provider_shap[i]:.4f} SHAP)*")
            else:
                st.error("SHAP values could not be computed for this provider.")

        # Tab 3: Model performance
        with tab3:
            st.subheader("Model Evaluation & Validation Metrics")
            metrics = artifact.get("metrics", {})

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("PR-AUC", f"{metrics.get('PR-AUC', 0):.2f}")
            m2.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.2f}")
            m3.metric("Recall", f"{metrics.get('Recall', 0):.2f}")
            m4.metric("Precision", f"{metrics.get('Precision', 0):.2f}")
            m5.metric("F1 Score", f"{metrics.get('F1', 0):.2f}")

            row_c1, row_c2 = st.columns(2)

            with row_c1:
                st.markdown("##### Test Set Confusion Matrix")
                y_true = test_preds["y_true"].values
                y_prob = test_preds["y_proba"].values
                y_curr_pred = (y_prob >= risk_threshold).astype(int)

                tn = int(np.sum((y_true == 0) & (y_curr_pred == 0)))
                fp = int(np.sum((y_true == 0) & (y_curr_pred == 1)))
                fn = int(np.sum((y_true == 1) & (y_curr_pred == 0)))
                tp = int(np.sum((y_true == 1) & (y_curr_pred == 1)))

                text_matrix = [
                    [f"TN — correctly cleared<br><b style='font-size:18px'>{tn:,}</b>",
                     f"FP — false alarm<br><b style='font-size:18px'>{fp:,}</b>"],
                    [f"FN — missed fraud ⚠️<br><b style='font-size:18px;color:#d9534f'>{fn:,}</b>",
                     f"TP — correctly caught<br><b style='font-size:18px'>{tp:,}</b>"],
                ]

                fig_cm = go.Figure()
                fig_cm.add_trace(go.Heatmap(
                    z=[[tn, fp], [fn, tp]],
                    x=["Predicted Non-Fraud", "Predicted Fraud"],
                    y=["Actual Non-Fraud", "Actual Fraud"],
                    text=text_matrix,
                    texttemplate="%{text}",
                    hoverinfo="text",
                    colorscale=[
                        [0.0, "#f0f7fb"],
                        [0.5, "#bedcf0"],
                        [1.0, "#3182bd"],
                    ],
                    showscale=False,
                ))

                fig_cm.add_shape(
                    type="rect",
                    x0=-0.48, x1=0.48,
                    y0=0.52, y1=1.48,
                    line=dict(color="#d9534f", width=3),
                    fillcolor="rgba(217, 83, 79, 0.22)",
                )

                fig_cm.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with row_c2:
                st.markdown("##### Precision-Recall Curve")
                prec, rec, _ = precision_recall_curve(y_true, y_prob)
                pr_auc_val = average_precision_score(y_true, y_prob)

                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=rec,
                    y=prec,
                    mode="lines",
                    name=f"PR Curve (AUC = {pr_auc_val:.3f})",
                    line=dict(color="#1f77b4", width=3),
                ))
                fig_pr.update_layout(
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_pr, use_container_width=True)

            shap_img_path = Path("reports/shap_summary.png")
            if shap_img_path.exists():
                st.markdown("##### Global Feature Importance (SHAP Summary)")
                st.image(str(shap_img_path), use_container_width=True)
            else:
                st.error("reports/shap_summary.png not found. Run python src/explain.py first.")

    # Tab 4: Batch Upload & Score
    with tab4:
        st.subheader("Upload raw claims data to score NEW providers")
        st.markdown(
            "Upload the same CSVs as Kaggle training data. The system will "
            "compute provider-level features automatically and score each provider."
        )

        # Sample template download expander
        with st.expander("📥 Download sample upload template (shows expected format)", expanded=False):
            st.write("Use these minimal reference CSVs to review column naming and structure:")
            c_t1, c_t2, c_t3 = st.columns(3)
            p_ip = Path("data/upload_template/sample_inpatient.csv")
            p_op = Path("data/upload_template/sample_outpatient.csv")
            p_bene = Path("data/upload_template/sample_beneficiary.csv")

            if p_ip.exists():
                with open(p_ip, "rb") as f:
                    c_t1.download_button("📥 Sample Inpatient CSV", data=f.read(), file_name="sample_inpatient.csv", mime="text/csv")
            if p_op.exists():
                with open(p_op, "rb") as f:
                    c_t2.download_button("📥 Sample Outpatient CSV", data=f.read(), file_name="sample_outpatient.csv", mime="text/csv")
            if p_bene.exists():
                with open(p_bene, "rb") as f:
                    c_t3.download_button("📥 Sample Beneficiary CSV", data=f.read(), file_name="sample_beneficiary.csv", mime="text/csv")

        st.markdown("##### Upload Claims & Beneficiary Datasets")
        u_col1, u_col2, u_col3 = st.columns(3)
        with u_col1:
            up_ip = st.file_uploader(
                "Inpatient Claims (Required)",
                type=["csv"],
                key="batch_inpatient_file",
                help="Train_Inpatientdata.csv format",
            )
        with u_col2:
            up_op = st.file_uploader(
                "Outpatient Claims (Optional)",
                type=["csv"],
                key="batch_outpatient_file",
                help="Train_Outpatientdata.csv format",
            )
        with u_col3:
            up_bene = st.file_uploader(
                "Beneficiary Data (Required)",
                type=["csv"],
                key="batch_beneficiary_file",
                help="Train_Beneficiarydata.csv format",
            )

        can_score = (up_ip is not None) and (up_bene is not None)
        if not can_score:
            st.info("👆 Please upload both **Inpatient Claims** and **Beneficiary Data** to enable provider scoring.")

        if st.button("🚀 Score Providers", disabled=not can_score, type="primary"):
            try:
                with st.spinner("Computing provider features and running ML inference..."):
                    df_ip = pd.read_csv(up_ip)
                    df_op = pd.read_csv(up_op) if up_op is not None else None
                    df_bene = pd.read_csv(up_bene)

                    batch_results = score_raw_claims(
                        inpatient_df=df_ip,
                        outpatient_df=df_op,
                        beneficiary_df=df_bene,
                        model_data=artifact,
                    )
                    st.session_state["batch_results"] = batch_results
                    st.success(f"Successfully processed raw claims and scored {len(batch_results):,} providers!")
            except Exception as err:
                st.error(f"Batch Scoring Error: {err}")

        # Render Batch Results if available
        if "batch_results" in st.session_state and st.session_state["batch_results"] is not None:
            res_df = st.session_state["batch_results"]
            if len(res_df) == 0:
                st.warning("No providers found in uploaded claims.")
            else:
                st.markdown("---")
                st.subheader("Batch Scoring Results")

                # KPI row
                k1, k2, k3 = st.columns(3)
                total_scored = len(res_df)
                high_count = int((res_df["risk_tier"] == "High").sum())
                avg_prob = float(res_df["fraud_probability"].mean())
                high_pct = (high_count / total_scored) * 100 if total_scored > 0 else 0

                k1.metric("Providers Scanned", f"{total_scored:,}")
                k2.metric("Flagged High Risk", f"{high_count:,}", f"↑ {high_pct:.1f}%")
                k3.metric("Average Risk Score", f"{avg_prob:.2%}")

                # Display table
                display_batch = res_df.copy()
                display_batch.insert(0, "Rank", range(1, len(display_batch) + 1))
                display_batch["fraud_probability"] = display_batch["fraud_probability"].apply(lambda v: f"{v:.4f}")

                def color_batch_tier(val):
                    if val == "High":
                        return "background-color: #ffcccc; color: #900C3F; font-weight: bold;"
                    elif val == "Medium":
                        return "background-color: #fff2cc; color: #B7950B; font-weight: bold;"
                    return "background-color: #d4edda; color: #155724;"

                styled_batch = display_batch.style.map(color_batch_tier, subset=["risk_tier"])
                st.dataframe(styled_batch, use_container_width=True)

                csv_res = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Results CSV",
                    data=csv_res,
                    file_name="batch_scored_providers.csv",
                    mime="text/csv",
                )

                st.caption("ℹ️ These providers were scored using the trained model. Verify with investigators before action.")

    # Tab 5: About & Architecture
    with tab5:
        st.subheader("System Architecture & Methodology")
        st.markdown(
            """
            ### Medicare Fraud Detection Pipeline Overview
            - **Dataset:** CMS Healthcare Provider Fraud Detection Analysis (5,410 Providers, 558,211 claims).
            - **Feature Engineering:** 30 domain-specific features capturing financial velocity, physician-beneficiary ratios, claim duration patterns, and clinical code entropy.
            - **Model Architecture:** XGBoost Classifier calibrated using stratified 5-fold cross-validation.
            - **Decision Threshold:** Balanced Strategy pinned at **0.28** (Precision ≥ 0.50), capturing **89.1%** of fraud with a 1-in-2 true alert precision.
            - **Explainability:** TreeSHAP attributions isolating top behavioural drivers per provider.
            """
        )

    # Footer disclaimer
    st.markdown("---")
    st.caption("⚖️ **Disclaimer:** Flags providers for review; not proof of fraud. Intended for investigational auditing assistance. Model trained on historical Medicare claims — performance on new geographies/providers not validated.")


if __name__ == "__main__":
    main()
