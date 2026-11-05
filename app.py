"""Streamlit interface for the trained length-of-stay classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/los_pipeline.joblib")

st.set_page_config(page_title="Healthcare LOS Analytics", page_icon="🏥", layout="wide")


@st.cache_resource
def load_artifact(path: str) -> dict:
    return joblib.load(path)


st.title("Patient Length-of-Stay Prediction")
st.caption("Educational machine-learning demonstration · Not for clinical decision-making")
st.warning(
    "This prototype is for learning and demonstration only. It is not medical advice "
    "and must not be used to make admission, treatment, or discharge decisions."
)

if not MODEL_PATH.is_file():
    st.info(
        "No trained model was found. Add the labeled training dataset at "
        "`data/raw/train.csv`, install the project dependencies, and run "
        "`python train.py` before starting the app."
    )
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH))
except Exception as exc:  # A readable message is preferable to a raw traceback in the UI.
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

pipeline = artifact["pipeline"]
feature_columns = artifact["feature_columns"]
schema = artifact["input_schema"]

st.subheader("Enter admission details")
with st.form("prediction_form"):
    inputs: dict[str, object] = {}
    left, right = st.columns(2)
    for index, column in enumerate(feature_columns):
        spec = schema[column]
        target = left if index % 2 == 0 else right
        label = column.replace("_", " ").strip().title()
        with target:
            if spec["kind"] == "categorical":
                options = spec["options"]
                default = spec["default"]
                selected_index = options.index(default) if default in options else 0
                inputs[column] = st.selectbox(label, options, index=selected_index)
            elif spec["kind"] == "text":
                inputs[column] = st.text_input(label, value=str(spec["default"]))
            else:
                kwargs = {
                    "label": label,
                    "min_value": float(spec["minimum"]),
                    "max_value": float(spec["maximum"]),
                    "value": float(spec["default"]),
                }
                if spec.get("integer"):
                    inputs[column] = st.number_input(
                        **kwargs,
                        step=1.0,
                        format="%.0f",
                    )
                else:
                    inputs[column] = st.number_input(**kwargs, step=0.1)

    submitted = st.form_submit_button("Predict length-of-stay category", type="primary")

if submitted:
    input_frame = pd.DataFrame([inputs], columns=feature_columns)
    try:
        prediction = str(pipeline.predict(input_frame)[0])
        st.success(f"Predicted length-of-stay category: **{prediction}**")
        st.caption("Prediction reflects patterns in the training dataset, not an individual clinical assessment.")
    except Exception as exc:
        st.error(f"Prediction failed. Check the saved model and input schema: {exc}")

with st.expander("Model evaluation from the saved training run"):
    metrics = artifact.get("metrics", {})
    if metrics:
        cols = st.columns(3)
        cols[0].metric("Holdout accuracy", f"{metrics.get('accuracy', 0):.1%}")
        cols[1].metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
        cols[2].metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.3f}")
        st.caption(
            f"Evaluated on {metrics.get('test_rows', '—')} holdout rows; "
            f"random state {metrics.get('random_state', 'recorded in training configuration')}."
        )
    else:
        st.write("No evaluation metrics were stored in this artifact.")
