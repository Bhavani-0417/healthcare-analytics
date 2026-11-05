import io

import joblib
import pandas as pd
import streamlit as st

from healthcare_los.modeling import predict_rows, train_evaluate

st.set_page_config(page_title="Healthcare LOS Analytics", page_icon="🏥", layout="wide")
st.title("Healthcare Analytics")
st.subheader("Patient Length-of-Stay Classification")
st.caption("Educational machine-learning baseline. Not for clinical decisions.")

with st.sidebar:
    upload = st.file_uploader("Upload labeled CSV", type=["csv"])
    test_size = st.slider("Hold-out test proportion", 0.10, 0.40, 0.20, 0.05)
    seed = st.number_input("Random seed", 0, 100000, 42)

if upload is None:
    st.info("Upload a labeled CSV to begin. The original project's target column was named Stay.")
    st.markdown(
        "**Workflow**\n1. Upload labeled data.\n2. Select the target.\n"
        "3. Train and evaluate.\n4. Upload new rows for prediction."
    )
    st.warning("Never upload identifiable patient information, confidential records, or data you are not authorized to use.")
    st.stop()

try:
    df = pd.read_csv(upload)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

a, b, c = st.columns(3)
a.metric("Rows", f"{len(df):,}")
b.metric("Columns", f"{len(df.columns):,}")
c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
st.dataframe(df.head(10), use_container_width=True)

default_target = "Stay" if "Stay" in df.columns else df.columns[-1]
target = st.selectbox("Target column", list(df.columns), index=list(df.columns).index(default_target))

if df[target].dropna().nunique() < 2:
    st.error("Target must contain at least two distinct classes.")
    st.stop()

if st.button("Train and evaluate model", type="primary"):
    try:
        with st.spinner("Training and evaluating..."):
            st.session_state["los_result"] = train_evaluate(
                df, target, float(test_size), int(seed)
            )
            st.session_state["los_target"] = target
    except Exception as exc:
        st.session_state.pop("los_result", None)
        st.error(f"Training failed: {exc}")

result = st.session_state.get("los_result")
if result is not None:
    if st.session_state.get("los_target") != target:
        st.info("Target changed. Train again to refresh the evaluation.")
        st.stop()

    st.markdown("### Hold-out evaluation")
    metrics = result["metrics"]
    cols = st.columns(4)
    for col, label, key in zip(
        cols,
        ["Accuracy", "Balanced accuracy", "Macro-F1", "Weighted-F1"],
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    ):
        col.metric(label, f"{metrics[key]:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Classification report")
        st.dataframe(result["classification_report"], use_container_width=True)
    with right:
        st.markdown("#### Confusion matrix")
        st.dataframe(result["confusion_matrix"], use_container_width=True)

    st.caption(
        f"Training rows: {result['n_train']:,} · "
        f"Hold-out rows: {result['n_test']:,} · Seed: {seed}"
    )

    buffer = io.BytesIO()
    joblib.dump(result["bundle"], buffer)
    st.download_button(
        "Download model bundle (.joblib)",
        buffer.getvalue(),
        "healthcare_los_model.joblib",
        "application/octet-stream",
    )

    st.markdown("### Predict new rows")
    prediction_file = st.file_uploader(
        "Upload an unlabeled CSV with the same feature columns",
        type=["csv"],
        key="prediction_csv",
    )
    if prediction_file is not None:
        try:
            new_rows = pd.read_csv(prediction_file)
            predictions = predict_rows(result["bundle"], new_rows)
            output = new_rows.copy()
            output["predicted_" + target] = predictions
            st.dataframe(output.head(20), use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                output.to_csv(index=False).encode("utf-8"),
                "healthcare_los_predictions.csv",
                "text/csv",
            )
        except Exception as exc:
            st.error(f"Could not generate predictions: {exc}")

    st.warning(
        "This is an educational baseline, not evidence of clinical safety "
        "or generalization to other hospitals."
    )
