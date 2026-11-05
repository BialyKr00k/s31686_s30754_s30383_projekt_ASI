"""Testy nodów data science."""

import pandas as pd

from asi_project.pipelines.data_science.nodes import split_data


def test_split_data_returns_train_and_test_parts(toy_contract, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "num": range(40),
            "cat": ["a", "b"] * 20,
            "label": ["yes", "no"] * 20,
        },
    )
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.load_project_config",
        lambda: type(
            "Config",
            (),
            {
                "model_features": ("num", "cat"),
                "numeric_features": ("num",),
                "categorical_features": ("cat",),
                "test_size": 0.2,
                "random_state": 42,
                "dataset_selected": True,
                "dataset_id": "toy",
            },
        )(),
    )
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.get_contract",
        lambda config: toy_contract,
    )

    X_train, X_test, y_train, y_test = split_data(
        frame,
        {"test_size": 0.2, "random_state": 42},
    )

    assert len(X_train) + len(X_test) == len(frame)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}
