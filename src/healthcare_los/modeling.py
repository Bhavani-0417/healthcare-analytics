"""Reusable preprocessing, model training, evaluation, and inference."""
from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def _build_pipeline(features: pd.DataFrame, random_state: int) -> Pipeline:
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   random_state=random_state, n_jobs=-1)
    return Pipeline([("preprocessor", prep), ("classifier", model)])

def train_evaluate(dataframe: pd.DataFrame, target_column: str = "Stay",
                   test_size: float = 0.20, random_state: int = 42) -> dict[str, Any]:
    """Split first, then fit imputers/encoders/model on training data only."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    if not 0.10 <= test_size <= 0.40:
        raise ValueError("test_size must be between 0.10 and 0.40.")
    clean = dataframe.dropna(subset=[target_column]).copy()
    if len(clean) < 10:
        raise ValueError("At least 10 rows with non-missing target are required.")
    if clean[target_column].nunique() < 2:
        raise ValueError("Target must contain at least two classes.")
    x = clean.drop(columns=[target_column])
    x = x.loc[:, x.notna().any(axis=0)]
    if x.empty:
        raise ValueError("No usable feature columns remain.")
    y = clean[target_column]
    counts = y.value_counts()
    n_classes = len(counts)
    n_test = int(round(len(clean) * test_size))
    n_train = len(clean) - n_test
    stratify = y if counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify)
    pipeline = _build_pipeline(x_train, random_state)
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)
    labels = list(pipeline.named_steps["classifier"].classes_)
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }
    bundle = {"model": pipeline, "target_column": target_column,
              "feature_columns": list(x.columns), "class_labels": [str(v) for v in labels],
              "metrics": metrics, "random_state": random_state}
    return {"bundle": bundle, "metrics": metrics,
            "classification_report": pd.DataFrame(report).transpose(),
            "confusion_matrix": pd.DataFrame(matrix, index=[str(v) for v in labels],
                                               columns=[str(v) for v in labels]),
            "n_train": len(x_train), "n_test": len(x_test)}

def predict_rows(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.Series:
    """Predict using a previously fitted preprocessing/model bundle."""
    if "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Model bundle is missing required fields.")
    columns = bundle["feature_columns"]
    missing = [c for c in columns if c not in rows.columns]
    if missing:
        raise ValueError("Input data is missing required feature columns: " + ", ".join(missing))
    return pd.Series(bundle["model"].predict(rows[columns]), name="prediction", index=rows.index)
