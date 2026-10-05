"""Cached, integrity-checked MovieLens downloads with atomic publication."""

import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

from sparkrec.artifacts import sha256_file, utc_now, write_json
from sparkrec.config import DatasetSettings

SOURCE_ROOT = "https://files.grouplens.org/datasets/movielens"
ALLOWED_FILES = {"ratings.csv", "movies.csv", "links.csv", "tags.csv", "README.txt"}


def verify_dataset(settings: DatasetSettings) -> dict:
    """Verify both the recorded archive identity and extracted file contents."""
    root = settings.input_dir
    with (root / "dataset.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("name") != settings.name:
        raise ValueError("Cached dataset name does not match configuration")
    if settings.archive_sha256 and manifest.get("archive_sha256") != settings.archive_sha256:
        raise ValueError("Cached archive checksum does not match configured archive_sha256")
    files = manifest.get("files", {})
    if not {"ratings.csv", "movies.csv", "README.txt"}.issubset(files):
        raise ValueError("Dataset manifest is missing required files")
    for name, checksum in files.items():
        if name not in ALLOWED_FILES or sha256_file(root / name) != checksum:
            raise ValueError(f"Cached dataset integrity check failed: {name}")
    archive = root.parent / f"{settings.name}.zip"
    if archive.exists() and sha256_file(archive) != manifest.get("archive_sha256"):
        raise ValueError("Cached ZIP checksum does not match dataset manifest")
    return manifest


def _extract(archive: Path, target: Path, dataset_name: str) -> list[str]:
    extracted = []
    with ZipFile(archive) as bundle:
        # Bound extraction to expected MovieLens files and reject unsafe paths.
        if sum(item.file_size for item in bundle.infolist()) > 10 * 1024**3:
            raise ValueError("Dataset archive exceeds the 10 GiB extraction limit")
        for item in bundle.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError(f"Unsafe archive member: {item.filename}")
            if item.is_dir():
                continue
            if len(path.parts) != 2 or path.parts[0] != dataset_name:
                raise ValueError(f"Unexpected archive layout: {item.filename}")
            name = path.name
            if name not in ALLOWED_FILES:
                continue
            if name in extracted:
                raise ValueError(f"Duplicate archive member: {name}")
            with bundle.open(item) as source, (target / name).open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(name)
    if not {"ratings.csv", "movies.csv", "README.txt"}.issubset(extracted):
        raise ValueError("MovieLens ZIP must contain ratings.csv, movies.csv, and README.txt")
    return sorted(extracted)


def download_dataset(settings: DatasetSettings, archive_path: Path | None = None) -> dict:
    if settings.name == "fixture":
        raise ValueError("The fixture is bundled with the project and does not need downloading")
    destination = settings.input_dir
    if destination.exists():
        # Never silently overwrite or adopt existing raw data.
        return verify_dataset(settings)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / f".{destination.name}.download.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(utc_now())
    try:
        archive = parent / f"{settings.name}.zip"
        source_url = f"{SOURCE_ROOT}/{settings.name}.zip"
        with tempfile.TemporaryDirectory(prefix=".download-", dir=parent) as temporary:
            staging = Path(temporary)
            downloaded = staging / "dataset.zip"
            if archive_path is not None:
                shutil.copyfile(archive_path, downloaded)
            elif archive.exists():
                shutil.copyfile(archive, downloaded)
            else:
                request = Request(source_url, headers={"User-Agent": "SparkRec/0.1"})
                with urlopen(request, timeout=60) as response, downloaded.open("wb") as output:
                    shutil.copyfileobj(response, output)
            checksum = sha256_file(downloaded)
            if settings.archive_sha256 and checksum != settings.archive_sha256:
                raise ValueError("Downloaded archive does not match configured archive_sha256")
            extracted = staging / "extracted"
            extracted.mkdir()
            names = _extract(downloaded, extracted, settings.name)
            manifest = {
                "schema_version": 1,
                "name": settings.name,
                "source_url": source_url,
                "acquired_at": utc_now(),
                "acquisition": "provided_archive" if archive_path else "download_or_cache",
                "archive_sha256": checksum,
                "license_file": "README.txt",
                "files": {name: sha256_file(extracted / name) for name in names},
            }
            write_json(extracted / "dataset.json", manifest)
            downloaded.replace(archive)
            extracted.rename(destination)
            return manifest
    finally:
        lock.unlink(missing_ok=True)
