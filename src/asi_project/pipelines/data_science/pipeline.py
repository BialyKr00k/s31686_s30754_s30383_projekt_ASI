"""Definicja pipeline'u trenowania i ewaluacji."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    """Buduje pipeline split -> trening -> ewaluacja."""
    return pipeline(
        [
            node(
                func=split_data,
                inputs={"prepared_data": "prepared_data", "split": "params:split"},
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_train_test_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs={
                    "model": "model",
                    "X_train": "X_train",
                    "X_test": "X_test",
                    "y_train": "y_train",
                    "y_test": "y_test",
                    "split": "params:split",
                },
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ],
    )
