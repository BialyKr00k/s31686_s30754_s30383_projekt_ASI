"""Testy artefaktu modelu i smoke testu (Sprint 3) — bez logowania do W&B i bez treningu."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from asi_project import model_artifact
from asi_project.model import make_dummy
from asi_project.model_artifact import (
    SmokeError,
    artifact_path,
    publish_package,
    smoke_check,
    smoke_rows,
    wandb_target,
)
from asi_project.model_package import save_model_package

FEATURES = ["num"]


def _package(tmp_path, metrics=None):
    """Mały pakiet z DummyClassifier zapisany tą samą funkcją, której używa trening."""
    model = make_dummy().fit(pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0]}), [0, 1, 0, 0])
    return save_model_package(
        tmp_path / "package",
        model,
        features=FEATURES,
        target="Churn",
        positive_class="1",
        metrics=metrics if metrics is not None else {"f1": 0.5},
        fingerprint="test-fingerprint",
        extra_metadata={"experiment": "dummy"},
    )


def _load_script(repo_root, name):
    spec = importlib.util.spec_from_file_location(name, repo_root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    """Model z kontrolowanymi wynikami, do sprawdzania, czy smoke wyłapuje złe odpowiedzi."""

    classes_ = np.array([0, 1])

    def __init__(self, predictions, proba):
        self._predictions, self._proba = predictions, proba

    def predict(self, X):
        return np.asarray(self._predictions)

    def predict_proba(self, X):
        return np.asarray(self._proba, dtype=float)


def _fake_package(monkeypatch, model):
    metadata = {
        "features": FEATURES,
        "target": "Churn",
        "positive_class": "1",
        "metrics": {"f1": 0.5},
        "git_sha": "abc1234",
        "fingerprint": "test-fingerprint",
        "library_versions": {"python": "3.11"},
    }
    monkeypatch.setattr(model_artifact, "load_model_package", lambda package_dir: (model, metadata))


def test_smoke_check_accepts_valid_package(tmp_path) -> None:
    rows = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0], "extra": list("abcde")})

    result, metadata = smoke_check(_package(tmp_path), rows)

    assert list(result.columns) == ["prediction", "proba_churn"]
    assert len(result) == 5
    assert set(result["prediction"]) <= {0, 1}
    assert result["proba_churn"].between(0, 1).all()
    assert metadata["experiment"] == "dummy"


def test_smoke_check_rejects_missing_feature(tmp_path) -> None:
    with pytest.raises(SmokeError, match="brakuje cech"):
        smoke_check(_package(tmp_path), pd.DataFrame({"other": [1.0]}))


def test_smoke_check_rejects_incomplete_metadata(tmp_path) -> None:
    with pytest.raises(SmokeError, match="metrics"):
        smoke_check(_package(tmp_path, metrics={}), pd.DataFrame({"num": [1.0]}))


@pytest.mark.parametrize(
    ("predictions", "proba", "message"),
    [
        ([0, 1], [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], "Liczba predykcji"),
        ([0, 2, 1], [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], "spoza klas"),
        ([0, 1, 1], [[1.0, 0.0], [0.0, 1.0]], "kształt"),
        ([0, 1, 1], [[0.9, 0.9], [0.0, 1.0], [0.5, 0.5]], "nie sumują"),
    ],
)
def test_smoke_check_rejects_wrong_response_schema(monkeypatch, predictions, proba, message):
    _fake_package(monkeypatch, FakeModel(predictions, proba))

    with pytest.raises(SmokeError, match=message):
        smoke_check("unused", pd.DataFrame({"num": [1.0, 2.0, 3.0]}))


def test_smoke_rows_are_normalized_sample_with_model_features(project) -> None:
    rows = smoke_rows(5)

    assert len(rows) == 5
    assert set(project.config.model_features) <= set(rows.columns)


def test_smoke_script_runs_on_local_package(project, repo_root, tmp_path, capsys) -> None:
    rows = smoke_rows(5)
    features = list(project.config.model_features)
    model = make_dummy().fit(rows.loc[:, features], [0, 1, 0, 0, 0])
    package_dir = save_model_package(
        tmp_path / "package",
        model,
        features=features,
        target=project.contract.target,
        positive_class=project.contract.positive_class,
        metrics={"f1": 0.0},
        fingerprint="test-fingerprint",
    )

    exit_code = _load_script(repo_root, "smoke_model").main([str(package_dir)])

    assert exit_code == 0
    assert "SMOKE: OK | 5 predykcji" in capsys.readouterr().out


def test_wandb_target_reads_parameters_and_allows_override(tmp_path, monkeypatch) -> None:
    params = tmp_path / "conf" / "base" / "parameters.yml"
    params.parent.mkdir(parents=True)
    params.write_text("experiments:\n  wandb:\n    entity: team\n    project: proj\n")
    monkeypatch.setenv("ASI_PROJECT_ROOT", str(tmp_path))

    assert wandb_target() == ("team", "proj")
    assert wandb_target(project="other") == ("team", "other")


def test_wandb_target_requires_entity_and_project(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ASI_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="entity"):
        wandb_target()


def test_publish_package_logs_artifact_with_production_alias(tmp_path, monkeypatch) -> None:
    wandb = pytest.importorskip("wandb")
    calls = {}

    class FakeArtifact:
        def __init__(self, name, type, description, metadata):
            calls["artifact"] = {"name": name, "type": type, "metadata": metadata}

        def add_dir(self, path):
            calls["dir"] = path

        def wait(self):
            return self

        version = "v0"

    class FakeRun:
        def log_artifact(self, artifact, aliases):
            calls["aliases"] = aliases
            return artifact

        def finish(self):
            calls["finished"] = True

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "Artifact", FakeArtifact)
    package_dir = _package(tmp_path)

    version = publish_package(package_dir, entity="team", project="proj")

    assert version == "team/proj/iranian-churn-model:v0"
    assert calls["aliases"] == ["production"]
    assert calls["artifact"]["type"] == "model"
    assert calls["artifact"]["metadata"]["fingerprint"] == "test-fingerprint"
    assert calls["dir"] == str(package_dir)
    assert calls["init"]["entity"] == "team" and calls["init"]["project"] == "proj"
    assert calls["finished"]


def test_publish_package_rejects_incomplete_metadata_before_wandb(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setattr(wandb, "init", lambda **kwargs: pytest.fail("nie wolno łączyć z W&B"))

    with pytest.raises(ValueError, match="metrics"):
        publish_package(_package(tmp_path, metrics={}), entity="team", project="proj")


def test_download_package_uses_production_alias(tmp_path, monkeypatch) -> None:
    wandb = pytest.importorskip("wandb")
    requested = {}

    class FakeApi:
        def artifact(self, name, type):
            requested["name"], requested["type"] = name, type
            return SimpleNamespace(download=lambda root: root)

    monkeypatch.setattr(wandb, "Api", FakeApi)

    path = model_artifact.download_package(entity="team", project="proj", root=tmp_path)

    assert requested == {"name": artifact_path("team", "proj"), "type": "model"}
    assert requested["name"] == "team/proj/iranian-churn-model:production"
    assert path == tmp_path
