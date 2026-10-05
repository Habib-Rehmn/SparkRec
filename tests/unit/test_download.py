import json
from io import BytesIO
from zipfile import ZipFile

import pytest

from sparkrec.artifacts import sha256_file
from sparkrec.config import DatasetSettings
from sparkrec.ingestion.download import download_dataset, verify_dataset


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "source.zip"
    with ZipFile(path, "w") as bundle:
        bundle.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n1,1,4,1\n")
        bundle.writestr("ml-latest-small/movies.csv", "movieId,title,genres\n1,Example,Drama\n")
        bundle.writestr("ml-latest-small/README.txt", "Synthetic test license")
    return path


def settings(tmp_path, checksum=None):
    return DatasetSettings(
        name="ml-latest-small",
        input_dir=tmp_path / "raw/ml-latest-small",
        snapshot_time="2020-01-01T00:00:00+00:00",
        archive_sha256=checksum,
    )


def test_archive_import_is_cached_and_verified(tmp_path, archive, monkeypatch):
    config = settings(tmp_path, sha256_file(archive))
    manifest = download_dataset(config, archive)
    monkeypatch.setattr(
        "sparkrec.ingestion.download.urlopen",
        lambda *a, **kw: pytest.fail("Cached download unexpectedly accessed network"),
    )
    assert download_dataset(config) == manifest
    assert manifest["archive_sha256"] == sha256_file(archive)
    assert (config.input_dir / "README.txt").read_text() == "Synthetic test license"
    (config.input_dir / "ratings.csv").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        verify_dataset(config)


def test_first_download_records_source_and_checksums(tmp_path, archive, monkeypatch):
    def response(request, timeout):
        assert request.full_url.endswith("/ml-latest-small.zip")
        assert timeout > 0
        return BytesIO(archive.read_bytes())

    monkeypatch.setattr("sparkrec.ingestion.download.urlopen", response)
    config = settings(tmp_path, sha256_file(archive))
    manifest = download_dataset(config)
    assert manifest == verify_dataset(config)
    assert manifest["source_url"] == (
        "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    )


def test_interrupted_download_does_not_publish_or_keep_lock(tmp_path, monkeypatch):
    class InterruptedResponse(BytesIO):
        def read(self, size=-1):
            raise OSError("connection interrupted")

    monkeypatch.setattr(
        "sparkrec.ingestion.download.urlopen", lambda *args, **kwargs: InterruptedResponse()
    )
    config = settings(tmp_path)
    with pytest.raises(OSError, match="interrupted"):
        download_dataset(config)
    assert not config.input_dir.exists()
    assert list(config.input_dir.parent.iterdir()) == []


def test_checksum_failure_does_not_publish(tmp_path, archive):
    config = settings(tmp_path, "0" * 64)
    with pytest.raises(ValueError, match="checksum|archive_sha256"):
        download_dataset(config, archive)
    assert not config.input_dir.exists()
    assert not list(config.input_dir.parent.glob("*.lock"))


def test_unsafe_archive_does_not_escape_or_publish(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped", "bad")
    config = settings(tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        download_dataset(config, archive)
    assert not (tmp_path / "escaped").exists()
    assert not config.input_dir.exists()


def test_incomplete_archive_is_rejected(tmp_path):
    archive = tmp_path / "incomplete.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("ml-latest-small/ratings.csv", "partial")
    with pytest.raises(ValueError, match="must contain"):
        download_dataset(settings(tmp_path), archive)


def test_cached_manifest_requires_expected_dataset(tmp_path, archive):
    config = settings(tmp_path)
    download_dataset(config, archive)
    path = config.input_dir / "dataset.json"
    manifest = json.loads(path.read_text())
    manifest["name"] = "another-dataset"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="name"):
        verify_dataset(config)
