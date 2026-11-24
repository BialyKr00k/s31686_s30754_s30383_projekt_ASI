"""Definicja pipeline'u eksperymentów: baseline + konfiguracje AutoGluona + tabela porównawcza."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_comparison_table, run_autogluon_experiment, run_baseline_experiment

# Nazwy muszą odpowiadać kluczom w conf/base/parameters.yml -> experiments.autogluon
AUTOGLUON_EXPERIMENTS = ("ag_gbm", "ag_trees_bagged")


def create_pipeline(**kwargs) -> Pipeline:
    """Buduje pipeline: baseline -> eksperymenty AutoGluona -> tabela porównawcza."""
    baseline = node(
        func=run_baseline_experiment,
        inputs={
            "model": "model",
            "X_test": "X_test",
            "y_test": "y_test",
            "split": "params:split",
            "experiments": "params:experiments",
        },
        outputs="baseline_result",
        name="baseline_experiment_node",
    )

    autogluon_nodes = [
        node(
            func=run_autogluon_experiment,
            inputs={
                "X_train": "X_train",
                "X_test": "X_test",
                "y_train": "y_train",
                "y_test": "y_test",
                "split": "params:split",
                "experiments": "params:experiments",
                "ag_config": f"params:experiments.autogluon.{name}",
            },
            outputs=f"{name}_result",
            name=f"{name}_experiment_node",
        )
        for name in AUTOGLUON_EXPERIMENTS
    ]

    comparison = node(
        func=build_comparison_table,
        inputs=["baseline_result", *[f"{name}_result" for name in AUTOGLUON_EXPERIMENTS]],
        outputs="model_comparison",
        name="build_comparison_table_node",
    )

    return pipeline([baseline, *autogluon_nodes, comparison])
