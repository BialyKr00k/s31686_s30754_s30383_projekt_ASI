"""Smoke test modelu poza CI (Sprint 3).

Ścieżka: pobranie artefaktu W&B z aliasem production → wczytanie modelu w tym (świeżym)
procesie, bez treningu → predykcja dla 5 wierszy próbki → kontrola schematu odpowiedzi.

Użycie:
    python scripts/smoke_model.py --wandb                          # artefakt production z W&B
    python scripts/smoke_model.py data/06_models/packages/ag_gbm   # pakiet lokalny, bez W&B
"""

import argparse
import sys
import tempfile
from pathlib import Path

from asi_project.model_artifact import (
    ARTIFACT_NAME,
    PRODUCTION_ALIAS,
    SmokeError,
    artifact_path,
    download_package,
    smoke_check,
    smoke_rows,
    wandb_target,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test pakietu modelu.")
    parser.add_argument("package_dir", nargs="?", type=Path, help="lokalny katalog pakietu")
    parser.add_argument("--wandb", action="store_true", help="pobierz artefakt z W&B")
    parser.add_argument("--entity", help="team W&B (domyślnie z conf/base/parameters.yml)")
    parser.add_argument("--project", help="projekt W&B (domyślnie z conf/base/parameters.yml)")
    parser.add_argument("--name", default=ARTIFACT_NAME, help="nazwa artefaktu w W&B")
    parser.add_argument("--alias", default=PRODUCTION_ALIAS, help="alias artefaktu w W&B")
    parser.add_argument("--rows", type=int, default=5, help="liczba wierszy do predykcji")
    args = parser.parse_args(argv)
    if args.wandb == (args.package_dir is not None):
        parser.error("podaj albo katalog pakietu, albo --wandb")

    rows = smoke_rows(args.rows)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        package_dir = args.package_dir
        if args.wandb:
            entity, project = wandb_target(args.entity, args.project)
            source = artifact_path(entity, project, args.name, args.alias)
            package_dir = download_package(
                entity=entity,
                project=project,
                name=args.name,
                alias=args.alias,
                root=Path(tmp) / "package",
            )
            print(f"Pobrano {source}")
        try:
            result, metadata = smoke_check(package_dir, rows)
        except (SmokeError, FileNotFoundError) as error:
            print(f"SMOKE: BŁĄD\n{error}", file=sys.stderr)
            return 1

    target = metadata["target"]
    if target in rows.columns:
        result.insert(0, f"{target} (prawda)", rows[target].to_numpy())
    print(result.to_string())
    print(
        f"SMOKE: OK | {len(result)} predykcji | model {metadata.get('experiment', '?')} "
        f"| F1 {metadata['metrics'].get('f1')} | git {metadata['git_sha'][:7]}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
