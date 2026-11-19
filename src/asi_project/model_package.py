"""Zapis i odczyt pakietu modelu używanego przez trening i API."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pandas


def git_sha() -> str:
    """Zwraca aktualny SHA commita."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def installed_versions() -> dict[str, str]:
    """Zwraca kluczowe wersje bibliotek."""
    names = (
        "pandas",
        "scikit-learn",
        "joblib",
        "kedro",
        "wandb",
        "autogluon.tabular",
    )
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            continue
    result["python"] = sys.version.split()[0]
    return result


def save_model_package(
    package_dir: str | Path,
    model: Any,
    *,
    features: list[str] | tuple[str, ...],
    target: str,
    positive_class: str,
    metrics: dict[str, Any],
    fingerprint: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Zapisuje model i wymagane metadata w katalogu pakietu."""
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    model_path = package_dir / "model.joblib"
    joblib.dump(model, model_path)

    metadata: dict[str, Any] = {
        "features": list(features),
        "target": target,
        "positive_class": str(positive_class),
        "metrics": metrics,
        "git_sha": git_sha(),
        "fingerprint": fingerprint,
        "library_versions": installed_versions(),
        "model_file": model_path.name,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    with (package_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return package_dir


def load_model_package(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Wczytuje model i metadata bez uruchamiania treningu."""
    package_dir = Path(package_dir)

    metadata_path = package_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    model_file = metadata.get("model_file", "model.joblib")
    model_path = package_dir / model_file
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    return joblib.load(model_path), metadata


def predict_from_package(
    package_dir: str | Path,
    rows: pandas.DataFrame,
) -> Any:
    """Wczytuje pakiet i wykonuje predykcję dla przekazanego schematu."""
    model, metadata = load_model_package(package_dir)
    features = metadata["features"]

    missing = [name for name in features if name not in rows.columns]
    if missing:
        raise ValueError(f"Brak wymaganych cech: {missing}")

    return model.predict(rows.loc[:, features])


# Proste aliasy dla innych modułów i przyszłego API.
save_model = save_model_package
load_model = load_model_package
