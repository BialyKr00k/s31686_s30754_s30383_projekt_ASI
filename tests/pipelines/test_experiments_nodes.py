"""Testy pipeline'u eksperymentów — bez dostępu do W&B i bez pełnego treningu."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from asi_project.model_package import load_model_package
from asi_project.pipeline_registry import register_pipelines
from asi_project.pipelines.experiments import nodes
from asi_project.pipelines.experiments.nodes import (
    COMPARISON_COLUMNS,
    build_comparison_table,
    run_autogluon_experiment,
    run_baseline_experiment,
)

SPLIT = {"test_size": 0.25, "random_state": 7}


@pytest.fixture
def experiments_params(tmp_path: Path) -> dict:
    return {
        "wandb": {"entity": None, "project": "test-project"},
        "packages_dir": str(tmp_path / "packages"),
        "autogluon_dir": str(tmp_path / "autogluon"),
        "measure_rows": 10,
        "measure_repeats": 2,
    }


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Testy nie łączą się z W&B, nie czytają project.yaml i nie startują nowych procesów."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(nodes, "_contract_info", lambda: ("Churn", "1"))
    monkeypatch.setattr(nodes, "_data_fingerprint", lambda: "test-fingerprint")
    monkeypatch.setattr(nodes, "_measure_memory_in_fresh_process", lambda package_dir: 1.5)
    monkeypatch.setattr(
        nodes,
        "_measure_ready_memory_in_fresh_process",
        lambda package_dir, sample: 2.5,
    )


@pytest.fixture
def toy_data():
    rows = 80
    X = pd.DataFrame({"num": [float(i) for i in range(rows)]})
    y = pd.Series([int(i > rows / 2) for i in range(rows)], name="Churn")
    return X.iloc[:60], X.iloc[60:], y.iloc[:60], y.iloc[60:]


def test_baseline_experiment_returns_full_row_and_package(experiments_params, toy_data) -> None:
    X_train, X_test, y_train, y_test = toy_data
    model = LogisticRegression(random_state=42).fit(X_train, y_train)

    row = run_baseline_experiment(model, X_test, y_test, SPLIT, experiments_params)

    assert row["model"] == "baseline_logreg"
    assert set(COMPARISON_COLUMNS) <= set(row)
    assert 0.0 <= row["f1"] <= 1.0
    assert row["predict_time_ms"] >= 0
    assert row["load_memory_mb"] == 1.5
    assert row["ready_memory_mb"] == 2.5

    _model, metadata = load_model_package(row["package_dir"])
    assert metadata["model_format"] == "joblib"
    assert metadata["fingerprint"] == "test-fingerprint"
    assert metadata["experiment"] == "baseline_logreg"


def test_autogluon_experiment_with_tiny_config(experiments_params, toy_data) -> None:
    pytest.importorskip("autogluon.tabular")
    X_train, X_test, y_train, y_test = toy_data
    ag_config = {
        "name": "ag_test",
        "presets": "medium_quality",
        "time_limit": 20,
        "hyperparameters": {"GBM": {}},
    }

    row = run_autogluon_experiment(
        X_train, X_test, y_train, y_test, SPLIT, experiments_params, ag_config
    )

    assert row["model"] == "ag_test"
    _model, metadata = load_model_package(row["package_dir"])
    assert metadata["model_format"] == "autogluon"
    assert metadata["params"]["time_limit"] == 20


def test_comparison_table_sorts_by_f1() -> None:
    rows = [
        {"model": "a", "f1": 0.5},
        {"model": "b", "f1": 0.9},
        {"model": "c", "f1": 0.7},
    ]

    table = build_comparison_table(*rows)

    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert list(table["model"]) == ["b", "c", "a"]


def test_default_pipeline_has_no_experiments() -> None:
    """`kedro run` (także w CI) nie może uruchamiać W&B ani AutoGluona."""
    pipelines = register_pipelines()

    default_nodes = {node.name for node in pipelines["__default__"].nodes}
    experiment_nodes = {node.name for node in pipelines["experiments"].nodes}

    assert not any("experiment" in name for name in default_nodes)
    assert default_nodes < experiment_nodes
    assert "baseline_experiment_node" in experiment_nodes
