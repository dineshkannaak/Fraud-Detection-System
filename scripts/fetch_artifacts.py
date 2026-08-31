"""Download and verify a model-artifact ZIP for deployment.

Environment variables:
    MODEL_ARTIFACT_URL: HTTPS URL for a ZIP containing model files.
    MODEL_ARTIFACT_SHA256: expected lowercase SHA-256 digest of the ZIP.
    MODEL_ARTIFACT_TOKEN: optional bearer token for private artifact storage.
    MODEL_DIR: destination directory, default /app/models.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def fetch() -> None:
    url = os.getenv("MODEL_ARTIFACT_URL")
    if not url:
        return
    if not url.lower().startswith("https://"):
        raise ValueError("MODEL_ARTIFACT_URL must use HTTPS")

    expected_hash = os.getenv("MODEL_ARTIFACT_SHA256", "").lower().strip()
    destination = Path(os.getenv("MODEL_DIR", "/app/models"))
    destination.mkdir(parents=True, exist_ok=True)
    token = os.getenv("MODEL_ARTIFACT_TOKEN")

    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        archive_path = Path(temporary.name)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as archive_file:
            shutil.copyfileobj(response, archive_file)

        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if expected_hash and digest != expected_hash:
            raise ValueError(f"Artifact checksum mismatch: expected {expected_hash}, got {digest}")

        with zipfile.ZipFile(archive_path) as archive:
            root = destination.resolve()
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not str(target).startswith(str(root) + os.sep):
                    raise ValueError("Artifact ZIP contains an unsafe path")
            archive.extractall(destination)
    finally:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    fetch()
