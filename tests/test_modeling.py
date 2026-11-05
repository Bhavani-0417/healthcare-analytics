import pandas as pd

from healthcare_los.modeling import build_input_schema, build_pipeline


def test_pipeline_handles_missing_and_unseen_categories():
    X_train = pd.DataFrame(
        {
            "age": [25, 40, None, 60, 35, 50],
            "department": ["A", "B", "A", "C", None, "B"],
        }
    )
    y_train = pd.Series(["short", "long", "short", "long", "short", "long"])

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    prediction = model.predict(pd.DataFrame({"age": [45], "department": ["new-category"]}))
    assert len(prediction) == 1
    assert prediction[0] in {"short", "long"}


def test_input_schema_has_numeric_and_categorical_fields():
    X = pd.DataFrame({"age": [20, 30, 40], "department": ["A", "B", "A"]})
    schema = build_input_schema(X)

    assert schema["age"]["kind"] == "numeric"
    assert schema["department"]["kind"] == "categorical"
    assert schema["department"]["default"] == "A"
