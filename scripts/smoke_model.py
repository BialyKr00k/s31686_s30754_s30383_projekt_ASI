"""Smoke test poza CI dla artefaktu production."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from asi_project.model_package import load_model_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    model, metadata = load_model_package(args.package_dir)
    features = metadata["features"]

    sample_path = Path("data/01_raw/sample.csv")
    if not sample_path.exists():
        print(f"Brak próbki: {sample_path}", file=sys.stderr)
        return 1

    rows = pd.read_csv(sample_path).loc[:, features].head(args.rows)
    predictions = model.predict(rows)

    if len(predictions) != args.rows:
        print("Niepoprawna liczba predykcji", file=sys.stderr)
        return 1

    print(f"OK: {len(predictions)} predykcji")
    print(f"Model: {metadata['git_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
