"""Strict configuration; relative paths resolve from the config's directory."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class OfflineSettings(Settings):
    output_dir: Path = Path("../artifacts/offline")
    validation_start: int | None = Field(default=None, gt=0)
    test_start: int | None = Field(default=None, gt=0)
    min_user_ratings: int = Field(default=5, ge=1)
    min_movie_ratings: int = Field(default=5, ge=1)
    ranks: list[int] = Field(default_factory=lambda: [20], min_length=1, max_length=4)
    reg_params: list[float] = Field(default_factory=lambda: [0.1], min_length=1, max_length=4)
    max_iter: int = Field(default=10, ge=1, le=100)
    seed: int = 42
    prior_weight: float = Field(default=10, gt=0, allow_inf_nan=False)
    relevance_threshold: float = Field(default=4, ge=0.5, le=5, allow_inf_nan=False)
    stored_k: int = Field(default=50, ge=20, le=500)
    initial_candidates: int = Field(default=200, ge=1)
    max_candidates: int = Field(default=2000, ge=1)
    audit_users: int = Field(default=5, ge=0, le=100)

    @model_validator(mode="after")
    def valid_protocol(self):
        if (self.validation_start is None) != (self.test_start is None):
            raise ValueError("Supply both temporal cutoffs or neither")
        if self.validation_start is not None and self.validation_start >= self.test_start:
            raise ValueError("validation_start must precede test_start")
        if any(rank < 1 or rank > 1000 for rank in self.ranks):
            raise ValueError("ALS ranks must be in 1..1000")
        import math

        if any(not math.isfinite(reg) or reg <= 0 for reg in self.reg_params):
            raise ValueError("ALS regularization must be finite and positive")
        if len(self.ranks) * len(self.reg_params) > 8:
            raise ValueError("Development search is capped at eight candidates")
        if self.initial_candidates > self.max_candidates:
            raise ValueError("initial_candidates must not exceed max_candidates")
        return self


class AppConfig(Settings):
    dataset: DatasetSettings
    output_dir: Path
    spark: SparkSettings = Field(default_factory=SparkSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    offline: OfflineSettings = Field(default_factory=OfflineSettings)


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
            "offline": config.offline.model_copy(
                update={"output_dir": (path.parent / config.offline.output_dir).resolve()}
            ),
        }
    )
