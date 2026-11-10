import hashlib
import json
import zipfile

import pytest

from app.bootstrap import bootstrap_model, install_bundle


CONFIG = {
    "patch_size": 256,
    "base_filters": 32,
    "mask_threshold": 0.5,
    "min_object_size": 12,
    "watershed_min_distance": 5,
}


def make_bundle(path):
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("best_unet.weights.h5", b"test-weights")
        bundle.writestr("frozen_config.json", json.dumps(CONFIG))


def test_install_bundle_atomically(tmp_path):
    bundle = tmp_path / "model.zip"
    model_dir = tmp_path / "model"
    make_bundle(bundle)
    expected = hashlib.sha256(bundle.read_bytes()).hexdigest()

    install_bundle(bundle, model_dir, expected)

    assert (model_dir / "best_unet.weights.h5").read_bytes() == b"test-weights"
    assert json.loads((model_dir / "frozen_config.json").read_text()) == CONFIG


def test_sha_mismatch_is_rejected(tmp_path):
    bundle = tmp_path / "model.zip"
    make_bundle(bundle)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        install_bundle(bundle, tmp_path / "model", "0" * 64)


def test_bootstrap_without_url_starts_setup_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("NUCLEI_MODEL_BUNDLE_URL", raising=False)
    assert "setup_required" in bootstrap_model(tmp_path)

