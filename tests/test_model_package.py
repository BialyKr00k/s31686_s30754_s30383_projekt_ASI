import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asi_project.model_package import (
    load_model_package,
    predict_from_package,
    save_model_package,
)


def test_model_package_save_load_predict(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    target = pd.Series([0, 0, 1, 1])

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42),
    )
    model.fit(frame, target)

    package_dir = save_model_package(
        tmp_path / "package",
        model,
        features=["num"],
        target="Churn",
        positive_class="1",
        metrics={"f1": 0.75},
        fingerprint="test-fingerprint",
    )

    loaded_model, metadata = load_model_package(package_dir)

    assert loaded_model is not None
    assert metadata["features"] == ["num"]
    assert metadata["target"] == "Churn"
    assert metadata["positive_class"] == "1"
    assert metadata["fingerprint"] == "test-fingerprint"
    assert metadata["git_sha"]
    assert metadata["library_versions"]

    predictions = predict_from_package(
        package_dir,
        pd.DataFrame({"num": [2.5, 3.5]}),
    )

    assert len(predictions) == 2


def test_model_package_reports_missing_features(tmp_path) -> None:
    frame = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
    target = pd.Series([0, 1, 1])

    model = LogisticRegression(random_state=42)
    model.fit(frame, target)

    package_dir = save_model_package(
        tmp_path / "package",
        model,
        features=["num"],
        target="Churn",
        positive_class="1",
        metrics={},
        fingerprint="test-fingerprint",
    )

    try:
        predict_from_package(package_dir, pd.DataFrame({"wrong": [1.0]}))
    except ValueError as error:
        assert "Brak wymaganych cech" in str(error)
    else:
        raise AssertionError("Nie wykryto brakujących cech")
