import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asi_project.model_package import (
    load_model_package,
    measure_load_memory_mb,
    measure_predict_time_ms,
    package_size_mb,
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


def test_package_size_counts_files_in_subfolders(tmp_path) -> None:
    package_dir = tmp_path / "package"
    (package_dir / "sub").mkdir(parents=True)

    one_mb = 1024 * 1024

    (package_dir / "a.bin").write_bytes(b"x" * one_mb)
    (package_dir / "sub" / "b.bin").write_bytes(b"x" * 2 * one_mb)

    assert package_size_mb(package_dir) == 3


def test_package_size_raises_for_missing_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        package_size_mb(tmp_path / "nie_istnieje")


class CountingModel:
    """Fałszywy model do testów — liczy wywołania predict. Duck typing"""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, rows):
        self.calls += 1
        return [0] * len(rows)


def test_measure_predict_time_does_warmup_and_repeats() -> None:
    model = CountingModel()
    rows = pd.DataFrame({"num": [1.0, 2.0, 3.0]})

    result = measure_predict_time_ms(model, rows, repeats=3)

    assert model.calls == 4
    assert isinstance(result, float)
    assert result >= 0


def test_measure_predict_time_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError):
        measure_predict_time_ms(CountingModel(), pd.DataFrame({"num": [1.0]}), repeats=0)


def test_measure_load_memory_returns_non_negative_mb(tmp_path) -> None:
    frame = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0]})
    target = pd.Series([0, 0, 1, 1])
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

    result = measure_load_memory_mb(package_dir)

    assert isinstance(result, float)
    assert result >= 0


def test_measure_load_memory_raises_for_missing_package(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        measure_load_memory_mb(tmp_path / "nie_istnieje")


def test_model_package_autogluon_roundtrip(tmp_path) -> None:
    tabular = pytest.importorskip("autogluon.tabular")

    frame = pd.DataFrame({"num": [float(i) for i in range(200)]})
    frame["Churn"] = (frame["num"] > 100).astype(int)

    predictor = tabular.TabularPredictor(
        label="Churn",
        path=str(tmp_path / "ag_train"),
        verbosity=0,
    )
    predictor.fit(frame, hyperparameters={"GBM": {}}, time_limit=30)

    package_dir = save_model_package(
        tmp_path / "package",
        predictor,
        features=["num"],
        target="Churn",
        positive_class="1",
        metrics={},
        fingerprint="test-fingerprint",
    )
    _loaded_model, metadata = load_model_package(package_dir)

    assert (package_dir / "model").is_dir()
    assert metadata["model_format"] == "autogluon"
    predictions = predict_from_package(package_dir, pd.DataFrame({"num": [5.0, 150.0]}))
    assert len(predictions) == 2
