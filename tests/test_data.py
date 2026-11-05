import pandas as pd
import pytest

from healthcare_los.data import prepare_features_target


def test_prepare_features_target_drops_identifiers_and_unlabeled_rows():
    frame = pd.DataFrame(
        {
            "case_id": [1, 2, 3],
            "patientid": [10, 20, 30],
            "age": [25, 41, 52],
            "department": ["A", "B", "A"],
            "Stay": ["0-10", None, "11-20"],
        }
    )

    X, y, dropped = prepare_features_target(frame)

    assert list(X.columns) == ["age", "department"]
    assert y.tolist() == ["0-10", "11-20"]
    assert dropped == ["case_id", "patientid"]


def test_prepare_features_target_requires_target():
    with pytest.raises(ValueError, match="Target column"):
        prepare_features_target(pd.DataFrame({"age": [30]}))
