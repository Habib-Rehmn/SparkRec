from pathlib import Path

import pytest
from pydantic import ValidationError

from sparkrec.cli import main
from sparkrec.config import AppConfig, load_config


def payload():
    return {
        "dataset": {
            "name": "fixture",
            "input_dir": "raw",
            "snapshot_time": "2020-01-01T00:00:00+00:00",
        },
        "output_dir": "output",
    }


@pytest.mark.parametrize("change", ["unknown_key", "bad_timezone", "bad_threshold"])
def test_rejects_invalid_configuration(change):
    data = payload()
    if change == "unknown_key":
        data["spark"] = {"shuffle_partitons": 2}
    elif change == "bad_timezone":
        data["dataset"]["snapshot_time"] = "2020-01-01T00:00:00"
    else:
        data["quality"] = {"max_rejected_fraction": 1.1}
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_paths_are_relative_to_configuration_file(tmp_path, monkeypatch):
    import yaml

    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(payload()))
    monkeypatch.chdir(Path("/tmp"))
    config = load_config(path)
    assert config.dataset.input_dir == tmp_path / "raw"
    assert config.output_dir == tmp_path / "output"


def test_rejects_overlapping_paths(tmp_path):
    import yaml

    data = payload()
    data["output_dir"] = "raw/output"
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="non-nested"):
        load_config(path)


def test_cli_configuration_error_is_nonzero(tmp_path, caplog):
    assert main(["validate-config", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "SparkRec failed" in caplog.text
