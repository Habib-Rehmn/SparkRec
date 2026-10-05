from pathlib import Path

import pytest

from sparkrec.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    return load_config(ROOT / "configs/fixture.yaml").model_copy(
        update={"output_dir": tmp_path / "runs"}
    )


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from sparkrec.config import SparkSettings
    from sparkrec.spark import spark_session

    with spark_session(
        SparkSettings(shuffle_partitions=2, log_level="ERROR"),
        tmp_path_factory.mktemp("spark"),
    ) as session:
        yield session
