"""Training, evaluation, and serialization for the LOS classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from healthcare_los.data import (
    DEFAULT_TARGET,
    load_training_data,
    prepare_features_target,
)


def _column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = features.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    numeric = [
        column for column in features.columns if column not in categorical
    ]
    return categorical, numeric


def build_pipeline(features: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """Build an end-to-end preprocessing and Random Forest classification pipeline."""
    categorical_columns, numeric_columns = _column_groups(features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_input_schema(features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build data-derived defaults for the Streamlit prediction form."""
    categorical_columns, numeric_columns = _column_groups(features)
    schema: dict[str, dict[str, Any]] = {}

    for column in categorical_columns:
        values = features[column].dropna().astype(str)
        options = sorted(values.unique().tolist())
        if not options:
            options = ["Unknown"]
        mode_values = values.mode()
        default = str(mode_values.iloc[0]) if not mode_values.empty else options[0]
        schema[column] = {
            "kind": "categorical" if len(options) <= 100 else "text",
            "options": options if len(options) <= 100 else [],
            "default": default,
        }

    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        if values.empty:
            minimum, maximum, default = 0.0, 1.0, 0.0
        else:
            minimum = float(values.min())
            maximum = float(values.max())
            default = float(values.median())
            if minimum == maximum:
                minimum -= 0.5
                maximum += 0.5
        schema[column] = {
            "kind": "numeric",
            "minimum": minimum,
            "maximum": maximum,
            "default": default,
            "integer": bool(pd.api.types.is_integer_dtype(features[column].dtype)),
        }

    return schema


def train_evaluate(
    dataframe: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a model on a holdout split; return its bundle and reports."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")

    features, target, dropped_ids = prepare_features_target(
        dataframe,
        target_column=target_column,
    )
    empty_columns = features.columns[features.isna().all()].tolist()
    if empty_columns:
        features = features.drop(columns=empty_columns)
    if features.empty:
        raise ValueError("No usable predictor columns remain after cleaning.")
    if len(features) < 10:
        raise ValueError("At least 10 labeled rows are required for training.")
    if target.nunique() < 2:
        raise ValueError("At least two target classes are required for classification.")

    class_counts = target.value_counts()
    n_classes = len(class_counts)
    n_test = int(round(len(features) * test_size))
    n_train = len(features) - n_test
    stratify = (
        target
        if class_counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes
        else None
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    pipeline = build_pipeline(x_train, random_state=random_state)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    labels = list(pipeline.named_steps["classifier"].classes_)
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
    }

    bundle = {
        "pipeline": pipeline,
        "feature_columns": list(features.columns),
        "input_schema": build_input_schema(x_train),
        "target_column": target_column,
        "excluded_identifier_columns": dropped_ids,
        "excluded_all_missing_columns": empty_columns,
        "metrics": metrics,
        "random_state": int(random_state),
    }
    return {
        "bundle": bundle,
        "metrics": metrics,
        "classification_report": pd.DataFrame(report).transpose(),
        "confusion_matrix": pd.DataFrame(
            matrix,
            index=[str(value) for value in labels],
            columns=[str(value) for value in labels],
        ),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }


def predict_rows(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.Series:
    """Predict rows with the saved preprocessing/model pipeline."""
    if "pipeline" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Model bundle is missing required fields.")

    columns = bundle["feature_columns"]
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(
            "Input data is missing required feature columns: " + ", ".join(missing)
        )
    return pd.Series(
        bundle["pipeline"].predict(rows[columns]),
        name="prediction",
        index=rows.index,
    )


def train_and_save(
    data_path: str | Path = "data/raw/train.csv",
    output_dir: str | Path = "artifacts",
    target_column: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train from a CSV, save the full model bundle, and write evaluation outputs."""
    frame = load_training_data(data_path)
    result = train_evaluate(
        frame,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["bundle"], destination / "los_pipeline.joblib")

    metrics = {
        **result["metrics"],
        "target_column": target_column,
        "training_rows": result["n_train"],
        "test_rows": result["n_test"],
        "feature_columns": result["bundle"]["feature_columns"],
        "identifier_columns_excluded": result["bundle"]["excluded_identifier_columns"],
        "all_missing_columns_excluded": result["bundle"]["excluded_all_missing_columns"],
        "random_state": int(random_state),
        "test_size": float(test_size),
        "classification_report": result["classification_report"].to_dict(),
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    result["confusion_matrix"].to_csv(
        destination / "confusion_matrix.csv",
        index_label="actual",
    )
    return metrics
