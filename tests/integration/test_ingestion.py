import json
import math
import shutil

import pytest
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import functions as F

from sparkrec.ingestion.pipeline import DataQualityError, ingest
from sparkrec.ingestion.validate import RATING_COLUMNS, read_raw, validate_ratings

pytestmark = pytest.mark.spark


def test_numeric_boundaries_are_quarantined(spark, tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,1,0.5,1\n"
        "2147483648,1,4,1\n"
        "0,1,4,1\n"
        "1,-1,4,1\n"
        "1,1,Infinity,1\n"
        "1,1,4.25,1\n"
        "1,1,4,0\n"
        "1,1,4,11\n"
    )
    movies = spark.createDataFrame([(1,)], "movie_id int")
    rows = validate_ratings(read_raw(spark, path, RATING_COLUMNS), movies, 10).collect()
    accepted = [row for row in rows if not row.reasons]
    assert [(row.user_id, row.movie_id, row.rating, row.timestamp) for row in accepted] == [
        (1, 1, 0.5, 1)
    ]
    rejected_reasons = [reason for row in rows for reason in row.reasons]
    assert rejected_reasons.count("invalid_user_id") == 2
    assert rejected_reasons.count("invalid_movie_id") == 1
    assert rejected_reasons.count("invalid_rating") == 2
    assert rejected_reasons.count("invalid_timestamp") == 2


def test_fixture_to_parquet_preserves_history_and_quarantines(spark, config):
    run = ingest(config, spark, "fixture-test")
    report = json.loads((run / "quality_report.json").read_text())
    assert report["ratings"]["total"] == 18
    assert report["ratings"]["rejected"] == 7
    assert report["ratings"]["retained"] == 10
    assert report["ratings"]["duplicate_events"] == 1
    assert report["ratings"]["repeated_user_movie_pairs"] == 1
    assert report["ratings"]["reason_counts"]["malformed_csv"] == 1
    ratings = spark.read.parquet(str(run / "ratings"))
    assert ratings.schema.simpleString() == (
        "struct<user_id:int,movie_id:int,rating:double,timestamp:bigint>"
    )
    updates = ratings.filter("user_id = 1 AND movie_id = 1").orderBy("timestamp").collect()
    assert [row.rating for row in updates] == [5.0, 4.5]
    assert spark.read.parquet(str(run / "quarantine/ratings")).count() == 7
    movies = spark.read.parquet(str(run / "movies"))
    assert movies.filter("movie_id = 4").first().genres == []
    assert movies.filter("movie_id = 5").first().title == 'A "Quoted" Title (2002)'
    assert json.loads((run / "manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        ingest(config, spark, "fixture-test")


def test_threshold_failure_writes_report_but_no_clean_outputs(spark, config):
    strict = config.model_copy(
        update={"quality": config.quality.model_copy(update={"max_rejected_fraction": 0.0})}
    )
    with pytest.raises(DataQualityError, match="exceeds"):
        ingest(strict, spark, "rejected")
    run = config.output_dir / "rejected"
    assert (run / "quality_report.json").exists()
    assert not (run / "ratings").exists()
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"


def test_conflicting_metadata_fails_before_ratings(spark, config, tmp_path):
    source = tmp_path / "input"
    shutil.copytree(config.dataset.input_dir, source)
    with (source / "movies.csv").open("a") as stream:
        stream.write("1,Contradictory title,Drama\n")
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"input_dir": source})}
    )
    with pytest.raises(DataQualityError, match="Conflicting"):
        ingest(config, spark, "conflicting")


def test_empty_input_fails_with_report(spark, config, tmp_path):
    source = tmp_path / "input"
    shutil.copytree(config.dataset.input_dir, source)
    (source / "ratings.csv").write_text("userId,movieId,rating,timestamp\n")
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"input_dir": source})}
    )
    with pytest.raises(DataQualityError, match="No usable ratings"):
        ingest(config, spark, "empty")


def test_als_runtime_and_model_roundtrip(spark, tmp_path):
    rows = [
        (user, movie, float((user + movie) % 5 + 1))
        for user in range(1, 5)
        for movie in range(1, 5)
    ]
    ratings = spark.createDataFrame(rows, "user int, item int, rating double")
    model = ALS(rank=2, maxIter=2, seed=42).fit(ratings)
    path = str(tmp_path / "model")
    model.save(path)
    restored = ALSModel.load(path)
    predictions = restored.transform(ratings).select("prediction").collect()
    assert len(predictions) == 16
    assert all(math.isfinite(row.prediction) for row in predictions)
    assert restored.recommendForAllUsers(2).filter(F.size("recommendations") == 2).count() == 4
