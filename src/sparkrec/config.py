"""Strict configuration; relative paths resolve from the config's directory."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetSettings(Settings):
    name: Literal["fixture", "ml-latest-small", "ml-32m"]
    input_dir: Path
    snapshot_time: datetime
    archive_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("snapshot_time")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot_time must include a timezone, for example +00:00")
        if value.timestamp() <= 0:
            raise ValueError("snapshot_time must be after the Unix epoch")
        return value


class SparkSettings(Settings):
    master: str = "local[2]"
    driver_memory: str = Field(default="1g", pattern=r"^[1-9][0-9]*[mg]$")
    shuffle_partitions: int = Field(default=4, ge=1, le=10000)
    log_level: Literal["ERROR", "WARN", "INFO"] = "WARN"


class QualitySettings(Settings):
    max_rejected_fraction: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)


class AppConfig(Settings):
    dataset: DatasetSettings
    output_dir: Path
    spark: SparkSettings = Field(default_factory=SparkSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)


def load_config(path: Path) -> AppConfig:
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        config = AppConfig.model_validate(yaml.safe_load(stream))
    input_dir = (path.parent / config.dataset.input_dir).resolve()
    output_dir = (path.parent / config.output_dir).resolve()
    if (
        input_dir == output_dir
        or input_dir in output_dir.parents
        or output_dir in input_dir.parents
    ):
        raise ValueError("input_dir and output_dir must be separate, non-nested directories")
    return config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"input_dir": input_dir}),
            "output_dir": output_dir,
        }
    )
