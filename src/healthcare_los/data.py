"""Dataset loading and feature/target preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TARGET = "Stay"
IDENTIFIER_COLUMNS = ("case_id", "patientid")


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load a labeled training CSV and validate its basic structure."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Training CSV not found: {csv_path}. Place the labeled training file at "
            "data/raw/train.csv or pass its location with --data."
        )

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"The training CSV is empty: {csv_path}")
    return frame


def prepare_features_target(
    frame: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate predictors and target, remove unlabeled rows and identifier columns."""
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )

    labeled = frame.dropna(subset=[target_column]).copy()
    if labeled.empty:
        raise ValueError(f"No rows have a non-null '{target_column}' target.")

    y = labeled[target_column].astype(str)
    X = labeled.drop(columns=[target_column])
    dropped_ids = [column for column in IDENTIFIER_COLUMNS if column in X.columns]
    if dropped_ids:
        X = X.drop(columns=dropped_ids)

    if X.empty:
        raise ValueError("No predictor columns remain after removing the target and identifiers.")

    return X.reset_index(drop=True), y.reset_index(drop=True), dropped_ids
