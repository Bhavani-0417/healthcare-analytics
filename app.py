"""Streamlit app for training and demonstrating a patient LOS classifier."""

from __future__ import annotations

import io

import joblib
import pandas as pd
import streamlit as st

from healthcare_los.modeling import predict_rows, train_evaluate

st.set_page_config(
    page_title="Healthcare LOS Analytics",
    page_icon="🏥",
    layout="wide",
)
st.title("Healthcare Analytics")
st.subheader("Patient Length-of-Stay Classification")
st.caption("Educational machine-learning baseline · Not for clinical decisions")

with st.sidebar:
    st.header("Experiment settings")
    upload = st.file_uploader("Upload labeled CSV", type=["csv"])
    test_size = st.slider("Hold-out test proportion", 0.10, 0.40, 0.20, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42)

st.warning(
    "This project is for educational use only. Never upload identifiable patient "
    "information, confidential records, or data you are not authorized to use. "
    "Predictions must not guide patient care, admission, or discharge."
)

if upload is None:
    st.info(
        "Upload a labeled CSV to begin. The original project's target column was named "
        "Stay. You can use data/synthetic_demo.csv for a demonstration; it contains "
        "artificial sample rows, not real patient records."
    )
    st.markdown(
        "**Workflow**\n"
        "1. Upload a labeled CSV.\n"
        "2. Select the target column.\n"
        "3. Train and review hold-out metrics.\n"
        "4. Download the model bundle.\n"
        "5. Upload new rows for batch prediction."
    )
    st.stop()

try:
    dataframe = pd.read_csv(upload)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if dataframe.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

first, second, third = st.columns(3)
first.metric("Rows", f"{len(dataframe):,}")
second.metric("Columns", f"{len(dataframe.columns):,}")
third.metric("Missing cells", f"{int(dataframe.isna().sum().sum()):,}")
st.dataframe(dataframe.head(10), use_container_width=True)

default_target = "Stay" if "Stay" in dataframe.columns else dataframe.columns[-1]
target_column = st.selectbox(
    "Target column",
    list(dataframe.columns),
    index=list(dataframe.columns).index(default_target),
)

if dataframe[target_column].dropna().nunique() < 2:
    st.error("The selected target must contain at least two distinct classes.")
    st.stop()

if st.button("Train and evaluate model", type="primary"):
    try:
        with st.spinner("Training and evaluating the model..."):
            st.session_state["los_result"] = train_evaluate(
                dataframe,
                target_column=target_column,
                test_size=float(test_size),
                random_state=int(seed),
            )
            st.session_state["los_target"] = target_column
    except Exception as exc:
        st.session_state.pop("los_result", None)
        st.error(f"Training failed: {exc}")

result = st.session_state.get("los_result")
if result is not None:
    if st.session_state.get("los_target") != target_column:
        st.info("The target selection changed. Train again to refresh the evaluation.")
        st.stop()

    st.markdown("### Hold-out evaluation")
    metrics = result["metrics"]
    metric_columns = st.columns(4)
    for column, label, key in zip(
        metric_columns,
        ["Accuracy", "Balanced accuracy", "Macro-F1", "Weighted-F1"],
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    ):
        column.metric(label, f"{metrics[key]:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Classification report")
        st.dataframe(result["classification_report"], use_container_width=True)
    with right:
        st.markdown("#### Confusion matrix")
        st.dataframe(result["confusion_matrix"], use_container_width=True)

    st.caption(
        f"Training rows: {result['n_train']:,} · "
        f"Hold-out rows: {result['n_test']:,} · Random seed: {seed}"
    )
    if result["bundle"]["excluded_identifier_columns"]:
        st.caption(
            "Excluded identifier fields: "
            + ", ".join(result["bundle"]["excluded_identifier_columns"])
        )

    buffer = io.BytesIO()
    joblib.dump(result["bundle"], buffer)
    st.download_button(
        "Download trained model (.joblib)",
        buffer.getvalue(),
        file_name="healthcare_los_model.joblib",
        mime="application/octet-stream",
    )

    st.markdown("### Predict new rows")
    prediction_upload = st.file_uploader(
        "Upload an unlabeled CSV with the same feature columns",
        type=["csv"],
        key="prediction_csv",
    )
    if prediction_upload is not None:
        try:
            new_rows = pd.read_csv(prediction_upload)
            predictions = predict_rows(result["bundle"], new_rows)
            output = new_rows.copy()
            output["predicted_" + target_column] = predictions
            st.dataframe(output.head(20), use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="healthcare_los_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not generate predictions: {exc}")

st.warning(
    "This result is a dataset-specific educational baseline, not evidence of clinical "
    "safety, fairness, or generalization to other hospitals."
)
