"""Registry de modelos: versões, concorrência, caminhos e integridade."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.registry import ModelRegistry


def _artifact(directory, name="modelo.joblib", content=b"artefato"):
    path = directory / name
    path.write_bytes(content)
    return path


def test_versions_are_sequential_per_model(tmp_path):
    registry = ModelRegistry(tmp_path)
    artifact = _artifact(tmp_path)
    assert [registry.register("a", artifact, {})["version"] for _ in range(3)] == [1, 2, 3]
    assert registry.register("b", artifact, {})["version"] == 1
    assert registry.latest("a")["alias"] == "a:v3"


def test_concurrent_registrations_never_reuse_a_version(tmp_path):
    artifact = _artifact(tmp_path)

    def register(_):
        # Um registry por thread, como instâncias independentes sobre o mesmo diretório.
        return ModelRegistry(tmp_path).register("modelo", artifact, {})["version"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        versions = list(pool.map(register, range(40)))
    assert sorted(versions) == list(range(1, 41))
    assert len(json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))) == 40
    assert not (tmp_path / "catalog.lock").exists()


def test_catalog_stores_only_the_file_name(tmp_path, monkeypatch):
    registry = ModelRegistry(tmp_path)
    record = registry.register("a", _artifact(tmp_path), {})
    assert record["artifact_path"] == "modelo.joblib"
    monkeypatch.chdir(tmp_path.parent)  # o registry não depende do diretório de trabalho
    assert registry.verify(record)
    assert registry.artifact_file(record) == tmp_path / "modelo.joblib"


def test_artifact_outside_registry_directory_is_refused(tmp_path):
    outside = tmp_path / "fora"
    outside.mkdir()
    registry = ModelRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match="inside the registry directory"):
        registry.register("a", _artifact(outside), {})


def test_catalog_entry_cannot_point_outside_registry(tmp_path):
    secret = _artifact(tmp_path, "segredo.bin", b"fora do registry")
    registry = ModelRegistry(tmp_path / "registry")
    forged = {"artifact_path": f"../{secret.name}", "checksum_sha256": "0" * 64}
    assert registry.artifact_file(forged).parent == registry.directory
    assert not registry.verify(forged)


def test_verify_detects_tampering(tmp_path):
    registry = ModelRegistry(tmp_path)
    artifact = _artifact(tmp_path)
    record = registry.register("a", artifact, {})
    assert registry.verify(record)
    artifact.write_bytes(b"adulterado")
    assert not registry.verify(record)


def test_signature_blocks_rewritten_checksum(tmp_path):
    """Quem escreve no diretório consegue refazer o checksum, mas não a assinatura HMAC."""
    signed = ModelRegistry(tmp_path, signing_key="segredo")
    artifact = _artifact(tmp_path)
    record = signed.register("a", artifact, {})
    assert signed.verify(record)

    artifact.write_bytes(b"pickle malicioso")
    forged = ModelRegistry(tmp_path).register("a", artifact, {})  # atacante sem a chave
    assert ModelRegistry(tmp_path).verify(forged), "sem chave, só o checksum é conferido"
    assert not signed.verify(forged)
    assert not signed.verify({**forged, "signature": record["signature"]})


def test_promotion_archives_previous_production(tmp_path):
    registry = ModelRegistry(tmp_path)
    artifact = _artifact(tmp_path)
    registry.register("a", artifact, {})
    registry.register("a", artifact, {})
    registry.promote("a", 1, "production")
    registry.promote("a", 2, "production", notes="melhor")
    assert registry.get("a", 1)["stage"] == "archived"
    assert registry.production("a")["version"] == 2
    with pytest.raises(KeyError):
        registry.promote("a", 99, "staging")
    with pytest.raises(ValueError, match="Invalid stage"):
        registry.promote("a", 1, "inexistente")
