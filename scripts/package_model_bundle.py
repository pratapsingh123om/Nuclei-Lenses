"""Package a trained checkpoint and frozen config for Render deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert args.weights.is_file(), f"Missing {args.weights}"
    assert args.config.is_file(), f"Missing {args.config}"
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert "mask_threshold" in config and "watershed_min_distance" in config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(args.weights, "best_unet.weights.h5")
        bundle.write(args.config, "frozen_config.json")
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"bundle": str(args.output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()

