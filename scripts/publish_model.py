"""Zapis wybranego pakietu modelu jako W&B Artifact z aliasem production (Sprint 3).

Najpierw sprawdza pakiet lokalnie (wczytanie → predykcja 5 wierszy → schemat odpowiedzi),
dopiero potem wysyła go do W&B. Entity i projekt W&B są brane z conf/base/parameters.yml
(experiments.wandb) albo z --entity / --project.

Użycie:
    python scripts/publish_model.py data/06_models/packages/ag_gbm --dry-run   # tylko sprawdzenie
    python scripts/publish_model.py data/06_models/packages/ag_gbm             # zapis do W&B
"""

import argparse
import sys
from pathlib import Path

from asi_project.model_artifact import (
    ARTIFACT_NAME,
    PRODUCTION_ALIAS,
    SmokeError,
    publish_package,
    smoke_check,
    smoke_rows,
    wandb_target,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zapis pakietu modelu jako W&B Artifact.")
    parser.add_argument("package_dir", type=Path, help="katalog pakietu (model + metadata.json)")
    parser.add_argument("--entity", help="team W&B (domyślnie z conf/base/parameters.yml)")
    parser.add_argument("--project", help="projekt W&B (domyślnie z conf/base/parameters.yml)")
    parser.add_argument("--name", default=ARTIFACT_NAME, help="nazwa artefaktu w W&B")
    parser.add_argument("--alias", default=PRODUCTION_ALIAS, help="alias nadawany tej wersji")
    parser.add_argument("--dry-run", action="store_true", help="tylko sprawdzenie, bez W&B")
    args = parser.parse_args(argv)

    try:
        result, metadata = smoke_check(args.package_dir, smoke_rows(5))
    except (SmokeError, FileNotFoundError) as error:
        print(f"Pakiet nie przeszedł sprawdzenia, nic nie wysyłam:\n{error}", file=sys.stderr)
        return 1
    print(f"Pakiet OK: {metadata.get('experiment', '?')} | F1 {metadata['metrics'].get('f1')}")
    print(result.to_string())

    if args.dry_run:
        print("--dry-run: nic nie zostało wysłane do W&B.")
        return 0

    entity, project = wandb_target(args.entity, args.project)
    version = publish_package(
        args.package_dir,
        entity=entity,
        project=project,
        name=args.name,
        aliases=(args.alias,),
    )
    print(f"Zapisano artefakt {version} z aliasem '{args.alias}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
