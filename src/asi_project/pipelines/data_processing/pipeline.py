"""Definicja pipeline'u przygotowania danych."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import filter_rows_for_training, normalize_raw_data, validate_contract


def create_pipeline(**kwargs) -> Pipeline:
    """Buduje pipeline: normalizacja wejścia, walidacja i filtrowanie treningowe."""
    return pipeline(
        [
            node(
                func=normalize_raw_data,
                inputs="raw_data",
                outputs="normalized_data",
                name="normalize_raw_data_node",
            ),
            node(
                func=validate_contract,
                inputs="normalized_data",
                outputs="validated_data",
                name="validate_contract_node",
            ),
            node(
                func=filter_rows_for_training,
                inputs="validated_data",
                outputs="prepared_data",
                name="filter_training_rows_node",
            ),
        ],
    )
