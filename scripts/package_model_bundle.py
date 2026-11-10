"""Package a trained checkpoint and frozen config for Render deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path


DEFAULT_ARTIFACTS = Path(
    os.getenv(
        "NUCLEI_ARTIFACTS_DIR",
        "/content/drive/MyDrive/Nuclei-lens/monuseg_project/artifacts",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package existing Nuclei Lens artifacts for Render; no training is performed."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="Drive artifact directory produced by the completed notebook.",
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    weights = args.weights or args.artifacts_dir / "best_unet.weights.h5"
    config_path = args.config or args.artifacts_dir / "frozen_config.json"
    output = args.output or args.artifacts_dir / "nuclei_lens_render_model.zip"

    assert weights.is_file(), f"Missing {weights}"
    assert config_path.is_file(), f"Missing {config_path}"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "mask_threshold" in config and "watershed_min_distance" in config
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(weights, "best_unet.weights.h5")
        bundle.write(config_path, "frozen_config.json")
    with zipfile.ZipFile(output) as bundle:
        assert set(bundle.namelist()) == {"best_unet.weights.h5", "frozen_config.json"}

    digest = sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    manifest = {
        "bundle": str(output),
        "bundle_bytes": output.stat().st_size,
        "sha256_file": str(checksum_path),
        "sha256": digest,
        "zip_members": ["best_unet.weights.h5", "frozen_config.json"],
        "render_environment": {
            "NUCLEI_MODEL_BUNDLE_SHA256": digest,
            "NUCLEI_MODEL_DIR": "/app/model",
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("\nNo training was run. Upload the ZIP, then set its HTTPS URL in Render as")
    print("NUCLEI_MODEL_BUNDLE_URL=https://YOUR-DOWNLOAD-URL/" + output.name)


if __name__ == "__main__":
    main()

