import joblib
import pandas as pd

from healthcare_los.modeling import build_input_schema, build_pipeline, train_and_save


def test_pipeline_handles_missing_and_unseen_categories():
    features = pd.DataFrame(
        {
            "age": [25, 40, None, 60, 35, 50],
            "department": ["A", "B", "A", "C", None, "B"],
        }
    )
    target = pd.Series(["short", "long", "short", "long", "short", "long"])

    model = build_pipeline(features)
    model.fit(features, target)

    prediction = model.predict(
        pd.DataFrame({"age": [45], "department": ["new-category"]})
    )
    assert len(prediction) == 1
    assert prediction[0] in {"short", "long"}


def test_input_schema_has_numeric_and_categorical_fields():
    features = pd.DataFrame(
        {"age": [20, 30, 40], "department": ["A", "B", "A"]}
    )
    schema = build_input_schema(features)

    assert schema["age"]["kind"] == "numeric"
    assert schema["department"]["kind"] == "categorical"
    assert schema["department"]["default"] == "A"


def test_training_saves_predictable_artifact_and_metrics(tmp_path):
    rows = 80
    frame = pd.DataFrame(
        {
            "case_id": range(rows),
            "patientid": range(1000, 1000 + rows),
            "age": [20 + (index % 55) for index in range(rows)],
            "department": [
                ["General", "Surgery", "Emergency"][index % 3]
                for index in range(rows)
            ],
            "Stay": [
                ["short", "medium", "long", "extended"][index % 4]
                for index in range(rows)
            ],
        }
    )
    dataset_path = tmp_path / "train.csv"
    output_path = tmp_path / "artifacts"
    frame.to_csv(dataset_path, index=False)

    metrics = train_and_save(dataset_path, output_path)
    artifact = joblib.load(output_path / "los_pipeline.joblib")
    sample = pd.DataFrame(
        [
            {
                column: artifact["input_schema"][column]["default"]
                for column in artifact["feature_columns"]
            }
        ]
    )

    assert metrics["test_rows"] == 16
    assert metrics["identifier_columns_excluded"] == ["case_id", "patientid"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(artifact["pipeline"].predict(sample)) == 1
    assert (output_path / "metrics.json").is_file()
    assert (output_path / "confusion_matrix.csv").is_file()
