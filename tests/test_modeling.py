import pandas as pd
import pytest

from healthcare_los.modeling import predict_rows, train_evaluate


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "case_id": list(range(100, 140)),
            "patientid": list(range(1000, 1040)),
            "age": list(range(20, 60)),
            "severity": ["low", "medium", "high", "low"] * 10,
            "admission_type": ["emergency", "elective"] * 20,
            "Stay": ["short", "long"] * 20,
        }
    )


def test_training_returns_metrics_and_excludes_identifiers(sample_data):
    result = train_evaluate(sample_data, random_state=7)

    assert set(result["metrics"]) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    }
    assert all(0 <= value <= 1 for value in result["metrics"].values())
    assert result["n_train"] + result["n_test"] == len(sample_data)
    assert result["bundle"]["target_column"] == "Stay"
    assert result["bundle"]["feature_columns"] == [
        "age",
        "severity",
        "admission_type",
    ]
    assert result["bundle"]["excluded_identifier_columns"] == [
        "case_id",
        "patientid",
    ]


def test_unseen_categories_are_supported(sample_data):
    result = train_evaluate(sample_data)
    rows = pd.DataFrame(
        {
            "age": [45],
            "severity": ["critical"],
            "admission_type": ["transfer"],
            "case_id": [999],
            "patientid": [9999],
        }
    )
    assert len(predict_rows(result["bundle"], rows)) == 1


def test_missing_features_are_rejected(sample_data):
    result = train_evaluate(sample_data)
    with pytest.raises(ValueError, match="missing required feature columns"):
        predict_rows(result["bundle"], pd.DataFrame({"age": [40]}))


def test_missing_target_rows_are_dropped(sample_data):
    sample_data.loc[0, "Stay"] = None
    result = train_evaluate(sample_data)
    assert result["n_train"] + result["n_test"] == len(sample_data) - 1


def test_missing_target_column_is_rejected(sample_data):
    with pytest.raises(ValueError, match="was not found"):
        train_evaluate(sample_data, target_column="unknown")
