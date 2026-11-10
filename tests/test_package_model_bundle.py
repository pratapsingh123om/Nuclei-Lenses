from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_zero_argument_layout_with_artifact_override(tmp_path):
    (tmp_path / "best_unet.weights.h5").write_bytes(b"weights")
    (tmp_path / "frozen_config.json").write_text(
        json.dumps({"mask_threshold": 0.5, "watershed_min_distance": 5}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_model_bundle.py"),
            "--artifacts-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    bundle = tmp_path / "nuclei_lens_render_model.zip"
    with zipfile.ZipFile(bundle) as handle:
        assert set(handle.namelist()) == {"best_unet.weights.h5", "frozen_config.json"}
    manifest = json.loads((tmp_path / "nuclei_lens_render_model.zip.json").read_text())
    assert manifest["sha256"]
    assert manifest["bundle_bytes"] > 0

