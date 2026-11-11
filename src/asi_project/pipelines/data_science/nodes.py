"""Nody pipeline'u data science."""

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import build_metrics_record, compute_metrics
from asi_project.model import build_baseline_pipeline, make_dummy


def split_data(
    prepared_data: pd.DataFrame,
    split: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Buduje X/y i wykonuje split zgodny z kontraktem datasetu."""
    config = load_project_config()
    contract = get_contract(config)
    X, y = make_xy(prepared_data, contract, config.model_features)
    return split_train_test(
        X,
        y,
        contract,
        test_size=float(split["test_size"]),
        random_state=int(split["random_state"]),
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Trenuje model razem z preprocessingiem sklearn."""
    config = load_project_config()
    model = build_baseline_pipeline(config.numeric_features, config.categorical_features)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split: dict[str, Any],
) -> dict[str, Any]:
    """Tworzy rekord metryk dla modelu logistycznego i dummy baseline'u."""
    config = load_project_config()
    contract = get_contract(config)
    dummy = make_dummy()
    dummy.fit(X_train, y_train)
    results = {
        "dummy": compute_metrics(y_test, dummy.predict(X_test)),
        "logistic_regression": compute_metrics(y_test, model.predict(X_test)),
    }
    record = build_metrics_record(
        contract=contract,
        config=config,
        features=config.model_features,
        n_train=len(X_train),
        n_test=len(X_test),
        data_mode="full",
        results=results,
    )
    record["split"]["test_size"] = float(split["test_size"])
    record["split"]["random_state"] = int(split["random_state"])
    return record
