"""Run-versioned ingestion; failed runs never replace successful output."""

import platform
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sparkrec import __version__
from sparkrec.artifacts import sha256_file, utc_now, write_json
from sparkrec.config import AppConfig
from sparkrec.ingestion.download import verify_dataset
from sparkrec.ingestion.validate import (
    MOVIE_COLUMNS,
    RATING_COLUMNS,
    quality_summary,
    read_raw,
    validate_movies,
    validate_ratings,
)


class DataQualityError(ValueError):
    """Inputs parsed, but cannot be published as a valid dataset."""


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def ingest(config: AppConfig, spark: SparkSession, run_id: str | None = None) -> Path:
    run_id = run_id or uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must contain 1-80 letters, digits, underscores, or hyphens")
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now(),
        "config": config.model_dump(mode="json"),
        "versions": {
            "sparkrec": __version__,
            "python": platform.python_version(),
            "spark": spark.version,
        },
        "git_revision": None,
    }
    write_json(run_dir / "manifest.json", manifest)
    cached = []
    try:
        try:
            manifest["git_revision"] = _git_revision()
        except FileNotFoundError:
            pass  # Git is optional in the runtime image.
        if config.dataset.name != "fixture":
            manifest["dataset"] = verify_dataset(config.dataset)
        manifest["input_sha256"] = {
            name: sha256_file(config.dataset.input_dir / name)
            for name in ("ratings.csv", "movies.csv")
        }
        write_json(run_dir / "manifest.json", manifest)

        movie_rows = validate_movies(
            read_raw(spark, config.dataset.input_dir / "movies.csv", MOVIE_COLUMNS)
        ).cache()
        cached.append(movie_rows)
        movies = movie_rows.filter(F.size("reasons") == 0).select("movie_id", "title", "genres")
        distinct_movies = movies.distinct()
        conflicts = distinct_movies.groupBy("movie_id").count().filter(F.col("count") > 1)
        movie_summary = quality_summary(movie_rows)
        movie_summary["duplicate_events"] = movies.count() - distinct_movies.count()
        movie_summary["conflicting_movie_ids"] = conflicts.count()
        movie_rows.filter(F.size("reasons") > 0).write.parquet(str(run_dir / "quarantine/movies"))
        report = {"movies": movie_summary}
        write_json(run_dir / "quality_report.json", report)
        if movie_summary["conflicting_movie_ids"]:
            conflicts.write.parquet(str(run_dir / "quarantine/movie_conflicts"))
            raise DataQualityError("Conflicting metadata for the same movie ID")
        if distinct_movies.isEmpty():
            raise DataQualityError("No usable movies remain")

        rating_rows = validate_ratings(
            read_raw(spark, config.dataset.input_dir / "ratings.csv", RATING_COLUMNS),
            distinct_movies,
            int(config.dataset.snapshot_time.timestamp()),
        ).cache()
        cached.append(rating_rows)
        accepted = rating_rows.filter(F.size("reasons") == 0).select(
            "user_id", "movie_id", "rating", "timestamp"
        )
        ratings = accepted.dropDuplicates().cache()
        cached.append(ratings)
        rating_summary = quality_summary(rating_rows)
        retained = ratings.count()
        rating_summary.update(
            {
                "retained": retained,
                "duplicate_events": rating_summary["accepted_before_deduplication"] - retained,
                "repeated_user_movie_pairs": ratings.groupBy("user_id", "movie_id")
                .count()
                .filter(F.col("count") > 1)
                .count(),
                "conflicting_same_timestamp_pairs": ratings.groupBy(
                    "user_id", "movie_id", "timestamp"
                )
                .agg(F.countDistinct("rating").alias("values"))
                .filter(F.col("values") > 1)
                .count(),
                "unique_users": ratings.select("user_id").distinct().count(),
                "unique_movies": ratings.select("movie_id").distinct().count(),
            }
        )
        movie_summary["retained"] = distinct_movies.count()
        report["ratings"] = rating_summary
        report["notes"] = [
            "Reason counts overlap; a record can have several validation errors.",
            "Only identical normalized events are removed. Updates are retained for M2 folds.",
        ]
        write_json(run_dir / "quality_report.json", report)
        rating_rows.filter(F.size("reasons") > 0).write.parquet(str(run_dir / "quarantine/ratings"))

        if retained == 0:
            raise DataQualityError("No usable ratings remain")
        for name, summary in (("movies", movie_summary), ("ratings", rating_summary)):
            if summary["rejected_fraction"] > config.quality.max_rejected_fraction:
                raise DataQualityError(
                    f"{name} rejected fraction {summary['rejected_fraction']:.4f} exceeds "
                    f"configured maximum {config.quality.max_rejected_fraction:.4f}"
                )

        distinct_movies.write.parquet(str(run_dir / "movies"))
        ratings.write.parquet(str(run_dir / "ratings"))
        manifest.update({"status": "complete", "outputs": ["movies", "ratings", "quarantine"]})
    except BaseException as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        manifest.update(
            {"finished_at": utc_now(), "elapsed_seconds": round(time.monotonic() - start, 3)}
        )
        write_json(run_dir / "manifest.json", manifest)
        for frame in reversed(cached):
            frame.unpersist()
    return run_dir
