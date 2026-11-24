"""Nody pipeline'u eksperymentów (Sprint 3).

Każdy eksperyment kończy się tak samo:
model -> metryki na X_test -> pakiet modelu -> pomiary (czas, rozmiar, pamięć) -> run W&B.
Dzięki temu baseline i wszystkie konfiguracje AutoGluona są porównywane identyczną metodą.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.metrics import compute_metrics
from asi_project.model_package import (
    git_sha,
    measure_predict_time_ms,
    package_size_mb,
    save_model_package,
)

# --- pomocnicze ---------------------------------------------------------------------------------


def _contract_info() -> tuple[str, str]:
    """Zwraca (target, klasa pozytywna) z kontraktu datasetu wybranego w project.yaml."""
    contract = get_contract(load_project_config())
    return contract.target, str(contract.positive_class)


def _data_fingerprint() -> str:
    """SHA-256 pliku danych z fingerprint.json (zapisuje go scripts/download_data.py)."""
    config = load_project_config()
    path = config.root / "data" / "01_raw" / str(config.dataset_id) / "fingerprint.json"
    if not path.is_file():
        return "unknown"
    return json.loads(path.read_text(encoding="utf-8"))["sha256"]


def _run_in_fresh_process(code: str) -> float:
    """Uruchamia kod w osobnym, świeżym procesie Pythona i zwraca liczbę z ostatniej linii wyjścia.

    W procesie, który przed chwilą trenował modele, pamięć jest „zabałaganiona”,
    więc pomiar byłby niewiarygodny. Świeży proces zaczyna od zera.
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip().splitlines()[-1])


def _measure_memory_in_fresh_process(package_dir: Path) -> float:
    """Pamięć (MB) zaraz po wczytaniu pakietu — w świeżym procesie."""
    code = (
        "from asi_project.model_package import measure_load_memory_mb\n"
        f"print(measure_load_memory_mb(r'{package_dir}'))"
    )
    return _run_in_fresh_process(code)


def _measure_ready_memory_in_fresh_process(package_dir: Path, sample: pd.DataFrame) -> float:
    """Pamięć (MB) po wczytaniu pakietu I pierwszej predykcji — w świeżym procesie.

    Niektóre biblioteki (np. AutoGluon) ładują modele leniwie: dopiero przy pierwszej
    predykcji. Ten pomiar pokazuje pamięć modelu „gotowego do odpowiadania”, tak jak w API.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = Path(tmp) / "sample.csv"
        sample.to_csv(sample_path, index=False)
        code = textwrap.dedent(
            f"""
            import gc

            import pandas as pd
            import psutil

            from asi_project.model_package import load_model_package

            rows = pd.read_csv(r'{sample_path}')
            process = psutil.Process()
            gc.collect()
            before = process.memory_info().rss
            model, _metadata = load_model_package(r'{package_dir}')
            model.predict(rows)
            after = process.memory_info().rss
            print(max(0.0, (after - before) / (1024 * 1024)))
            """
        )
        return _run_in_fresh_process(code)


def _log_to_wandb(
    wandb_params: dict[str, Any],
    run_name: str,
    run_config: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Zapisuje jeden eksperyment jako osobny run W&B.

    Tryb pracy W&B ustawia zmienna środowiskowa WANDB_MODE
    (np. ``disabled`` w testach i CI — wtedy nic nie jest wysyłane).
    """
    import wandb

    run = wandb.init(
        entity=wandb_params.get("entity"),
        project=wandb_params["project"],
        name=run_name,
        job_type="experiment",
        config=run_config,
    )
    run.log(results)
    run.summary.update(results)
    run.finish()


def _evaluate_package_and_log(
    *,
    run_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_params: dict[str, Any],
    experiments: dict[str, Any],
) -> dict[str, Any]:
    """Wspólna ścieżka dla każdego modelu: metryki -> pakiet -> pomiary -> W&B -> wiersz tabeli."""
    target, positive_class = _contract_info()
    fingerprint = _data_fingerprint()

    # 1. metryki na zbiorze testowym (klasa pozytywna = 1)
    predictions = pd.Series(model.predict(X_test)).astype(int)
    metrics = compute_metrics(y_test, predictions)

    # 2. pakiet modelu — ta sama funkcja, której użyje API
    package_dir = save_model_package(
        Path(experiments["packages_dir"]) / run_name,
        model,
        features=list(X_test.columns),
        target=target,
        positive_class=positive_class,
        metrics=metrics,
        fingerprint=fingerprint,
        extra_metadata={"experiment": run_name, "params": run_params},
    )

    # 3. pomiary operacyjne — identyczne dla każdego modelu
    sample = X_test.head(int(experiments["measure_rows"]))
    measurements = {
        "predict_time_ms": measure_predict_time_ms(
            model,
            sample,
            repeats=int(experiments["measure_repeats"]),
        ),
        "package_size_mb": package_size_mb(package_dir),
        "load_memory_mb": _measure_memory_in_fresh_process(package_dir),
        "ready_memory_mb": _measure_ready_memory_in_fresh_process(package_dir, sample),
    }

    # 4. run W&B: parametry, metryki, Git SHA, fingerprint danych
    scalar_metrics = {name: metrics[name] for name in ("accuracy", "precision", "recall", "f1")}
    _log_to_wandb(
        experiments["wandb"],
        run_name,
        run_config={
            **run_params,
            "git_sha": git_sha(),
            "data_fingerprint": fingerprint,
            "n_test": len(X_test),
        },
        results={**scalar_metrics, **measurements},
    )

    # 5. jeden wiersz tabeli porównawczej
    return {"model": run_name, **scalar_metrics, **measurements, "package_dir": str(package_dir)}


# --- nody ---------------------------------------------------------------------------------------


def run_baseline_experiment(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    split: dict[str, Any],
    experiments: dict[str, Any],
) -> dict[str, Any]:
    """Loguje baseline z poprzednich sprintów (LogisticRegression) jako pierwszy run W&B.

    Model nie jest trenowany ponownie ani zmieniany — bierzemy dokładnie ten,
    który wytrenował pipeline data_science.
    """
    return _evaluate_package_and_log(
        run_name="baseline_logreg",
        model=model,
        X_test=X_test,
        y_test=y_test,
        run_params={
            "model": "LogisticRegression",
            "test_size": float(split["test_size"]),
            "random_state": int(split["random_state"]),
        },
        experiments=experiments,
    )


def run_autogluon_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split: dict[str, Any],
    experiments: dict[str, Any],
    ag_config: dict[str, Any],
) -> dict[str, Any]:
    """Trenuje jedną konfigurację AutoGluona (z limitem czasu) i przepuszcza ją
    przez tę samą ścieżkę co baseline: metryki -> pakiet -> pomiary -> run W&B."""
    from autogluon.tabular import TabularPredictor

    run_name = ag_config["name"]
    label = str(y_train.name)

    # AutoGluon chce jednej tabeli: cechy + kolumna z targetem
    train_data = X_train.copy()
    train_data[label] = y_train.to_numpy()

    predictor = TabularPredictor(
        label=label,
        problem_type="binary",
        eval_metric="f1",
        path=str(Path(experiments["autogluon_dir"]) / run_name),
        verbosity=1,
    )
    predictor.fit(
        train_data,
        presets=ag_config["presets"],
        time_limit=int(ag_config["time_limit"]),
        hyperparameters=ag_config.get("hyperparameters"),
    )

    return _evaluate_package_and_log(
        run_name=run_name,
        model=predictor,
        X_test=X_test,
        y_test=y_test,
        run_params={
            "model": "AutoGluon",
            "presets": ag_config["presets"],
            "time_limit": int(ag_config["time_limit"]),
            "hyperparameters": ag_config.get("hyperparameters"),
            "best_model": predictor.model_best,
            "test_size": float(split["test_size"]),
            "random_state": int(split["random_state"]),
        },
        experiments=experiments,
    )


COMPARISON_COLUMNS = (
    "model",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "predict_time_ms",
    "package_size_mb",
    "load_memory_mb",
    "ready_memory_mb",
    "package_dir",
)


def build_comparison_table(*results: dict[str, Any]) -> pd.DataFrame:
    """Skleja wiersze wszystkich eksperymentów w jedną tabelę (najlepsze F1 na górze)."""
    table = pd.DataFrame(list(results), columns=list(COMPARISON_COLUMNS))
    return table.sort_values("f1", ascending=False).reset_index(drop=True)
