"""Testy nodów data science."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from asi_project.config import ProjectConfig
from asi_project.metrics import validate_metrics_record
from asi_project.pipelines.data_science.nodes import evaluate_model, split_data, train_model

NODES = "asi_project.pipelines.data_science.nodes"
SPLIT_PARAMS = {"test_size": 0.25, "random_state": 7}


@pytest.fixture
def toy_config(tmp_path: Path) -> ProjectConfig:
    """Konfiguracja projektu z innym splitem niż params, żeby wykryć mylenie źródeł."""
    return ProjectConfig(
        root=tmp_path,
        dataset_id="toy",
        numeric_features=("num",),
        categorical_features=("cat",),
        test_size=0.5,
        random_state=0,
        ci_sample_path=tmp_path / "sample.csv",
        metrics_path=tmp_path / "metrics.json",
    )


@pytest.fixture
def patched_config(monkeypatch, toy_config, toy_contract) -> ProjectConfig:
    monkeypatch.setattr(f"{NODES}.load_project_config", lambda: toy_config)
    monkeypatch.setattr(f"{NODES}.get_contract", lambda config: toy_contract)
    return toy_config


@pytest.fixture
def prepared_frame() -> pd.DataFrame:
    rows = 40
    return pd.DataFrame(
        {
            "num": [float(i) for i in range(rows)],
            "cat": ["a", "b"] * (rows // 2),
            "label": ["yes", "no"] * (rows // 2),
        },
    )


@pytest.fixture
def split_parts(patched_config, prepared_frame):
    return split_data(prepared_frame, SPLIT_PARAMS)


def test_split_data_returns_train_and_test_parts(split_parts, prepared_frame) -> None:
    X_train, X_test, y_train, y_test = split_parts

    assert len(X_train) + len(X_test) == len(prepared_frame)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}


def test_split_data_uses_params_not_project_config(split_parts, prepared_frame) -> None:
    _, X_test, _, _ = split_parts

    assert len(X_test) == int(len(prepared_frame) * SPLIT_PARAMS["test_size"])


def test_split_data_is_deterministic(patched_config, prepared_frame) -> None:
    first = split_data(prepared_frame, SPLIT_PARAMS)
    second = split_data(prepared_frame, SPLIT_PARAMS)

    pd.testing.assert_index_equal(first[1].index, second[1].index)


def test_split_data_keeps_only_model_features(split_parts) -> None:
    X_train, X_test, _, _ = split_parts

    assert list(X_train.columns) == ["num", "cat"]
    assert list(X_test.columns) == ["num", "cat"]


def test_train_model_returns_fitted_sklearn_pipeline(split_parts) -> None:
    X_train, X_test, y_train, _ = split_parts

    model = train_model(X_train, y_train)

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_train_model_handles_unseen_category_and_missing_values(split_parts) -> None:
    X_train, _, y_train, _ = split_parts
    model = train_model(X_train, y_train)
    new_rows = pd.DataFrame({"num": [None, 3.0], "cat": ["nowa", None]})

    predictions = model.predict(new_rows)

    assert len(predictions) == 2


def test_evaluate_model_returns_valid_metrics_record(split_parts, toy_contract) -> None:
    X_train, X_test, y_train, y_test = split_parts
    model = train_model(X_train, y_train)

    record = evaluate_model(model, X_train, X_test, y_train, y_test, SPLIT_PARAMS)

    assert validate_metrics_record(record, toy_contract) == []
    assert set(record["models"]) == {"dummy", "logistic_regression"}
    assert record["split"]["n_train"] == len(X_train)
    assert record["split"]["n_test"] == len(X_test)


def test_evaluate_model_records_split_params(split_parts) -> None:
    X_train, X_test, y_train, y_test = split_parts
    model = train_model(X_train, y_train)

    record = evaluate_model(model, X_train, X_test, y_train, y_test, SPLIT_PARAMS)

    assert record["split"]["test_size"] == SPLIT_PARAMS["test_size"]
    assert record["split"]["random_state"] == SPLIT_PARAMS["random_state"]


def test_pipeline_wires_nodes_catalog_and_params() -> None:
    from asi_project.pipelines.data_science import create_pipeline

    pipe = create_pipeline()

    assert [node.name for node in pipe.nodes] == [
        "split_train_test_node",
        "train_model_node",
        "evaluate_model_node",
    ]
    assert {"prepared_data", "params:split"} <= pipe.inputs()
    assert {"model", "metrics"} <= pipe.all_outputs()
