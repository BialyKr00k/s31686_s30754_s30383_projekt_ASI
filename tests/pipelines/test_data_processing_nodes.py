"""Testy nodów przygotowania danych."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from asi_project.data import DataContractError
from asi_project.pipelines.data_processing import create_pipeline
from asi_project.pipelines.data_processing.nodes import (
    filter_rows_for_training,
    normalize_raw_data,
    validate_contract,
)

NODES = "asi_project.pipelines.data_processing.nodes"


@pytest.fixture
def toy_project(toy_contract, monkeypatch):
    """Podmienia project.yaml i kontrakt na syntetyczne (cechy: num, cat)."""
    config = SimpleNamespace(
        model_features=("num", "cat"),
        numeric_features=("num",),
        categorical_features=("cat",),
    )
    monkeypatch.setattr(f"{NODES}.load_project_config", lambda: config)
    monkeypatch.setattr(f"{NODES}.get_contract", lambda config: toy_contract)
    return config


def test_normalize_raw_data_cleans_headers_and_text() -> None:
    data = pd.DataFrame({"Call  Failure": [" 1 "], "Status": [" Active "]})

    normalized = normalize_raw_data(data)

    assert list(normalized.columns) == ["Call Failure", "Status"]
    assert normalized.loc[0, "Status"] == "Active"


def test_validate_contract_passes_valid_data_unchanged(toy_project, toy_frame) -> None:
    validated = validate_contract(toy_frame)

    pd.testing.assert_frame_equal(validated, toy_frame)


def test_validate_contract_rejects_missing_column(toy_project, toy_frame) -> None:
    with pytest.raises(DataContractError):
        validate_contract(toy_frame.drop(columns="cat"))


def test_validate_contract_rejects_excluded_feature(toy_project, toy_frame) -> None:
    # kolumna EXCLUDED zadeklarowana jako cecha modelu musi zatrzymać pipeline
    toy_project.model_features = ("num", "cat", "leak")

    with pytest.raises(DataContractError):
        validate_contract(toy_frame)


def test_filter_rows_for_training_drops_duplicates_and_missing_target(
    toy_project,
    toy_frame,
) -> None:
    data = pd.concat([toy_frame, toy_frame.head(3)], ignore_index=True)
    data.loc[5, "label"] = np.nan

    filtered = filter_rows_for_training(data)

    assert len(filtered) == len(toy_frame) - 1
    assert not filtered.duplicated().any()
    assert filtered["label"].notna().all()
    pd.testing.assert_frame_equal(filtered, data.loc[filtered.index])


def test_pipeline_goes_from_raw_to_prepared_data() -> None:
    pipeline = create_pipeline()

    assert pipeline.inputs() == {"raw_data"}
    assert pipeline.outputs() == {"prepared_data"}
    assert [node.name for node in pipeline.nodes] == [
        "normalize_raw_data_node",
        "validate_contract_node",
        "filter_training_rows_node",
    ]
