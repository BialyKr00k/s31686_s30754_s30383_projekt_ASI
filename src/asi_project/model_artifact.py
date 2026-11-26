"""Pakiet modelu w W&B: zapis jako Artifact z aliasem i pobranie po aliasie (Sprint 3).

Do W&B trafia ten sam katalog, który robi ``save_model_package`` (model + ``metadata.json``).
Dalsza część systemu (smoke modelu, API w Sprincie 4) pobiera model po aliasie
``production``, a nie po numerze eksperymentu.

W&B jest importowane dopiero w funkcjach, które go potrzebują, więc ``smoke_check``
i testy w CI działają bez logowania do W&B.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from asi_project.config import find_repo_root, load_project_config
from asi_project.data import label_text, normalize_column_names, normalize_input
from asi_project.model_package import load_model_package

ARTIFACT_NAME = "iranian-churn-model"
ARTIFACT_TYPE = "model"
PRODUCTION_ALIAS = "production"
REQUIRED_METADATA = (
    "features",
    "target",
    "positive_class",
    "metrics",
    "git_sha",
    "fingerprint",
    "library_versions",
)
ALLOWED_LABELS = {"0", "1"}  # make_xy koduje target: klasa pozytywna -> 1, druga klasa -> 0


class SmokeError(RuntimeError):
    """Pakiet modelu nie przeszedł smoke testu (wczytanie, predykcja albo schemat)."""


# --- gdzie w W&B -----------------------------------------------------------------------------


def wandb_target(entity: str | None = None, project: str | None = None) -> tuple[str, str]:
    """Entity i projekt W&B: z argumentów albo z ``conf/base/parameters.yml`` (experiments.wandb)."""
    if entity is None or project is None:
        path = find_repo_root() / "conf" / "base" / "parameters.yml"
        params = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        settings = ((params or {}).get("experiments") or {}).get("wandb") or {}
        entity = entity or settings.get("entity")
        project = project or settings.get("project")
    if not entity or not project:
        raise ValueError(
            "Brak entity albo projektu W&B: podaj --entity i --project "
            "albo uzupełnij experiments.wandb w conf/base/parameters.yml.",
        )
    return entity, project


def artifact_path(
    entity: str,
    project: str,
    name: str = ARTIFACT_NAME,
    alias: str = PRODUCTION_ALIAS,
) -> str:
    """Pełna nazwa artefaktu w W&B, np. ``team/projekt/iranian-churn-model:production``."""
    return f"{entity}/{project}/{name}:{alias}"


# --- kontrola pakietu ----------------------------------------------------------------------------


def read_metadata(package_dir: str | Path) -> dict[str, Any]:
    """Czyta ``metadata.json`` pakietu (bez wczytywania modelu)."""
    path = Path(package_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_problems(metadata: dict[str, Any]) -> list[str]:
    """Brakujące albo puste pola wymagane w ``metadata.json`` (pusta lista = OK)."""
    missing = [key for key in REQUIRED_METADATA if not metadata.get(key)]
    return [f"metadata.json: brak pola '{key}'." for key in missing]


def smoke_rows(n_rows: int = 5) -> pd.DataFrame:
    """Pierwsze wiersze próbki CI po input normalization, tak jak dane wejściowe w API."""
    config = load_project_config()
    if not config.ci_sample_path.is_file():
        raise FileNotFoundError(f"Brak próbki {config.ci_sample_path}.")
    rows = normalize_input(normalize_column_names(pd.read_csv(config.ci_sample_path)))
    return rows.head(n_rows).reset_index(drop=True)


def _positive_proba(model: Any, X: pd.DataFrame) -> tuple[np.ndarray | None, list[str]]:
    """Prawdopodobieństwo klasy pozytywnej (1) i lista problemów ze schematem ``predict_proba``."""
    raw = model.predict_proba(X)
    if isinstance(raw, pd.DataFrame):  # AutoGluon: kolumny to klasy
        classes, values = list(raw.columns), raw.to_numpy(dtype=float)
    else:  # sklearn: kolejność kolumn jak w model.classes_
        classes, values = list(getattr(model, "classes_", [0, 1])), np.asarray(raw, dtype=float)
    if values.shape != (len(X), 2):
        return None, [f"predict_proba ma kształt {values.shape}, oczekiwano {(len(X), 2)}."]
    problems = []
    if not np.allclose(values.sum(axis=1), 1.0) or ((values < 0) | (values > 1)).any():
        problems.append("predict_proba: wartości spoza [0, 1] albo wiersze nie sumują się do 1.")
    labels = [label_text(name) for name in classes]
    if "1" not in labels:
        problems.append(f"predict_proba: brak kolumny klasy pozytywnej 1 (klasy: {labels}).")
        return None, problems
    return values[:, labels.index("1")], problems


def smoke_check(
    package_dir: str | Path,
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Wczytuje pakiet, robi predykcję dla ``rows`` i sprawdza schemat odpowiedzi.

    Sprawdza: pola ``metadata.json``, obecność cech modelu w danych, liczbę predykcji,
    etykiety tylko 0/1 oraz poprawne ``predict_proba``. Przy problemie zgłasza ``SmokeError``.
    Zwraca tabelkę (``prediction``, ``proba_churn``) i metadane pakietu.
    """
    model, metadata = load_model_package(package_dir)
    problems = metadata_problems(metadata)
    features = list(metadata.get("features") or [])
    missing = [name for name in features if name not in rows.columns]
    if missing:
        problems.append(f"W danych brakuje cech modelu: {missing}.")
    if problems:
        raise SmokeError("\n".join(problems))

    X = rows.loc[:, features]
    predictions = np.asarray(model.predict(X))
    if len(predictions) != len(X):
        problems.append(f"Liczba predykcji {len(predictions)} zamiast {len(X)}.")
    labels = {label_text(value) for value in predictions}
    if not labels <= ALLOWED_LABELS:
        problems.append(f"Predykcje spoza klas 0/1: {sorted(map(str, labels))}.")
    proba, proba_problems = _positive_proba(model, X)
    problems.extend(proba_problems)
    if problems:
        raise SmokeError("\n".join(problems))

    result = pd.DataFrame({"prediction": predictions.astype(int), "proba_churn": proba})
    return result, metadata


# --- W&B -----------------------------------------------------------------------------------------


def publish_package(
    package_dir: str | Path,
    *,
    entity: str,
    project: str,
    name: str = ARTIFACT_NAME,
    aliases: tuple[str, ...] = (PRODUCTION_ALIAS,),
) -> str:
    """Zapisuje katalog pakietu jako W&B Artifact (typ ``model``) z podanymi aliasami.

    Alias w W&B wskazuje zawsze jedną wersję artefaktu: po ponownym zapisie z aliasem
    ``production`` alias przechodzi na nową wersję. Zwraca nazwę zapisanej wersji.
    """
    import wandb

    package_dir = Path(package_dir)
    metadata = read_metadata(package_dir)
    problems = metadata_problems(metadata)
    if problems:
        raise ValueError("\n".join(problems))

    run = wandb.init(
        entity=entity,
        project=project,
        name=f"publish-{metadata.get('experiment', 'model')}",
        job_type="model-selection",
        config={
            "experiment": metadata.get("experiment"),
            "git_sha": metadata["git_sha"],
            "data_fingerprint": metadata["fingerprint"],
            "aliases": list(aliases),
        },
    )
    try:
        artifact = wandb.Artifact(
            name=name,
            type=ARTIFACT_TYPE,
            description="Pakiet modelu produkcyjnego: model + metadata.json (Sprint 3).",
            metadata=metadata,
        )
        artifact.add_dir(str(package_dir))
        logged = run.log_artifact(artifact, aliases=list(aliases))
        logged.wait()
        version = logged.version
    finally:
        run.finish()
    return f"{entity}/{project}/{name}:{version}"


def download_package(
    *,
    entity: str,
    project: str,
    name: str = ARTIFACT_NAME,
    alias: str = PRODUCTION_ALIAS,
    root: str | Path | None = None,
) -> Path:
    """Pobiera artefakt modelu z W&B po aliasie i zwraca katalog pakietu."""
    import wandb

    artifact = wandb.Api().artifact(artifact_path(entity, project, name, alias), type=ARTIFACT_TYPE)
    return Path(artifact.download(root=None if root is None else str(root)))
