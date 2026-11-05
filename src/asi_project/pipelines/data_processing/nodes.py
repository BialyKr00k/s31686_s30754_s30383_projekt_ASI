"""Nody pipeline'u przygotowania danych."""

import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import (
    assert_valid,
    filter_training_rows,
    normalize_column_names,
    normalize_input,
)


def normalize_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje nagłówki i wartości wejściowe przed walidacją kontraktu."""
    return normalize_input(normalize_column_names(raw_data))


def validate_contract(data: pd.DataFrame) -> pd.DataFrame:
    """Waliduje dane względem kontraktu wybranego w project.yaml."""
    config = load_project_config()
    assert_valid(data, get_contract(config), config)
    return data


def filter_rows_for_training(data: pd.DataFrame) -> pd.DataFrame:
    """Usuwa rekordy, których nie można użyć do treningu."""
    config = load_project_config()
    return filter_training_rows(data, get_contract(config))
