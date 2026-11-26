"""Rejestr pipeline'ów Kedro."""

from kedro.pipeline import Pipeline

from asi_project.pipelines import data_processing, data_science, experiments


def register_pipelines() -> dict[str, Pipeline]:
    """Rejestruje pipeline'y projektu.

    Pipeline'y są importowane jawnie (a nie przez ``find_pipelines``), bo od Kedro 1.5
    ``kedro run --pipelines <nazwa>`` ładuje przez ``find_pipelines`` tylko pipeline
    o tej nazwie — a ``experiments`` musi zawierać też przygotowanie danych i trening.

    - ``__default__`` (czyli ``kedro run``, także w CI) to tylko przygotowanie danych
      i baseline — bez AutoGluona i bez W&B.
    - ``experiments`` (``kedro run --pipelines experiments``) to pełna ścieżka:
      dane -> baseline -> eksperymenty -> W&B.
      Zawiera data_processing i data_science, bo eksperymenty potrzebują
      X_train / X_test / model, które Kedro trzyma tylko w pamięci jednego uruchomienia.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    data_science_pipeline = data_science.create_pipeline()
    training = data_processing_pipeline + data_science_pipeline

    return {
        "__default__": training,
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "experiments": training + experiments.create_pipeline(),
    }
