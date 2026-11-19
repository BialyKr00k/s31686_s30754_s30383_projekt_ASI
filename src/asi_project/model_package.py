"""Zapis i odczyt pakietu modelu używanego przez trening i API."""

from __future__ import annotations

import gc
import json
import statistics
import subprocess
import sys
import time
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


def package_size_mb(package_dir: str | Path) -> float:
    """Zwraca łączny rozmiar wszystkich plików w katalogu pakietu w MB."""
    package_dir = Path(package_dir)

    if not package_dir.is_dir():
        raise FileNotFoundError(package_dir)

    total_bytes = 0

    for item in package_dir.rglob("*"):
        if item.is_file():
            total_bytes += item.stat().st_size

    return total_bytes / (1024 * 1024)


def measure_predict_time_ms(model: Any, rows: pandas.DataFrame, repeats: int = 5) -> float:
    """Zwraca medianę czasu predykcji (ms) dla podanych wierszy, po jednej rozgrzewce."""
    if repeats < 1:
        raise ValueError("repeats musi być >= 1")

    model.predict(rows)

    times_ms = []

    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(rows)
        elapsed_s = time.perf_counter() - start
        times_ms.append(elapsed_s * 1000)

    return statistics.median(times_ms)


def measure_load_memory_mb(package_dir: str | Path) -> float:
    """Zwraca przyrost pamięci RAM procesu (MB) po wczytaniu pakietu modelu."""
    import psutil

    process = psutil.Process()
    gc.collect()

    before = process.memory_info().rss
    _model, _metadata = load_model_package(package_dir)
    after = process.memory_info().rss

    return max(0.0, (after - before) / (1024 * 1024))


def _is_autogluon(model: Any) -> bool:
    """Sprawdza, czy model to predyktor AutoGluona (bez importowania autogluon)."""
    return type(model).__module__.startswith("autogluon.tabular")


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

    if _is_autogluon(model):
        model_path = package_dir / "model"
        model.clone(path=str(model_path), return_clone=False, dirs_exist_ok=True)
        model_format = "autogluon"
    else:
        model_path = package_dir / "model.joblib"
        joblib.dump(model, model_path)
        model_format = "joblib"

    metadata: dict[str, Any] = {
        "features": list(features),
        "target": target,
        "positive_class": str(positive_class),
        "metrics": metrics,
        "git_sha": git_sha(),
        "fingerprint": fingerprint,
        "library_versions": installed_versions(),
        "model_file": model_path.name,
        "model_format": model_format,
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

    model_format = metadata.get("model_format", "joblib")
    model_file = metadata.get("model_file", "model.joblib")
    model_path = package_dir / model_file

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if model_format == "autogluon":
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(model_path))
    else:
        model = joblib.load(model_path)

    return model, metadata


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
