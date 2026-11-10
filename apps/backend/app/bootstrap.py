from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_FILES = ("best_unet.weights.h5", "frozen_config.json")
REQUIRED_CONFIG_KEYS = {
    "patch_size",
    "base_filters",
    "mask_threshold",
    "min_object_size",
    "watershed_min_distance",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_bundle(bundle_path: Path, model_dir: Path, expected_sha256: str = "") -> None:
    if expected_sha256:
        actual = file_sha256(bundle_path)
        if actual.lower() != expected_sha256.strip().lower():
            raise RuntimeError(f"Model bundle SHA-256 mismatch: expected {expected_sha256}, received {actual}")

    model_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path) as bundle:
        names = set(bundle.namelist())
        missing = sorted(set(REQUIRED_FILES) - names)
        if missing:
            raise RuntimeError(f"Model bundle is missing: {', '.join(missing)}")

        config = json.loads(bundle.read("frozen_config.json"))
        missing_keys = sorted(REQUIRED_CONFIG_KEYS - set(config))
        if missing_keys:
            raise RuntimeError(f"frozen_config.json is missing keys: {', '.join(missing_keys)}")

        temporary_paths = []
        try:
            for filename in REQUIRED_FILES:
                target = model_dir / filename
                temporary = model_dir / f".{filename}.download"
                with bundle.open(filename) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if temporary.stat().st_size == 0:
                    raise RuntimeError(f"Model bundle contains an empty {filename}")
                temporary_paths.append((temporary, target))
            for temporary, target in temporary_paths:
                try:
                    os.replace(temporary, target)
                except PermissionError:
                    # Some managed Windows filesystems deny rename even within
                    # one directory. Render/Linux uses the atomic path above;
                    # this fallback preserves compatibility for local tests.
                    shutil.copyfile(temporary, target)
                    temporary.unlink()
        finally:
            for temporary, _ in temporary_paths:
                temporary.unlink(missing_ok=True)


def download_bundle(url: str, destination: Path, max_bytes: int) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError("NUCLEI_MODEL_BUNDLE_URL must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "nuclei-lens-render/1.0"})
    downloaded = 0
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise RuntimeError("Model bundle exceeds NUCLEI_MODEL_BUNDLE_MAX_MB")
            output.write(chunk)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise RuntimeError("NUCLEI_MODEL_BUNDLE_GCS_URI must start with gs://")
    bucket, separator, object_name = uri[5:].partition("/")
    if not separator or not bucket or not object_name:
        raise RuntimeError("GCS model URI must include both bucket and object name")
    return bucket, object_name


def download_gcs_bundle(uri: str, destination: Path, max_bytes: int) -> None:
    # Imported lazily so local setup_required mode does not require credentials.
    from google.cloud import storage

    bucket_name, object_name = parse_gcs_uri(uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    blob.reload()
    if blob.size is None or blob.size <= 0:
        raise RuntimeError(f"GCS model object is empty: {uri}")
    if blob.size > max_bytes:
        raise RuntimeError("GCS model bundle exceeds NUCLEI_MODEL_BUNDLE_MAX_MB")
    blob.download_to_filename(str(destination), checksum="auto")


def bootstrap_model(model_dir: Path) -> str:
    if all((model_dir / filename).is_file() for filename in REQUIRED_FILES):
        return "existing model artifacts found"

    gcs_uri = os.getenv("NUCLEI_MODEL_BUNDLE_GCS_URI", "").strip()
    bundle_url = os.getenv("NUCLEI_MODEL_BUNDLE_URL", "").strip()
    if not gcs_uri and not bundle_url:
        return "model artifacts absent; starting in setup_required mode"

    maximum_mb = max(1, int(os.getenv("NUCLEI_MODEL_BUNDLE_MAX_MB", "1024")))
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=model_dir, suffix=".zip", delete=False) as handle:
        temporary_bundle = Path(handle.name)
    try:
        if gcs_uri:
            download_gcs_bundle(gcs_uri, temporary_bundle, max_bytes=maximum_mb * 1024 * 1024)
        else:
            download_bundle(bundle_url, temporary_bundle, max_bytes=maximum_mb * 1024 * 1024)
        install_bundle(
            temporary_bundle,
            model_dir,
            expected_sha256=os.getenv("NUCLEI_MODEL_BUNDLE_SHA256", ""),
        )
    finally:
        temporary_bundle.unlink(missing_ok=True)
    source = gcs_uri if gcs_uri else bundle_url
    return f"model bundle downloaded and installed from {source}"

