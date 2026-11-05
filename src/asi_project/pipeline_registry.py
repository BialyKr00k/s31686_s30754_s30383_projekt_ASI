"""Rejestr pipeline'ów Kedro."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Rejestruje pipeline'y z ``src/asi_project/pipelines/``.

    ``find_pipelines`` znajduje każdy podpakiet z funkcją ``create_pipeline``
    (``data_processing``, ``data_science``). ``__default__`` (czyli ``kedro run``)
    to suma wszystkich znalezionych pipeline'ów — Kedro ustala kolejność nodów
    na podstawie wejść i wyjść z Data Catalogu.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values(), start=Pipeline([]))
    return pipelines
