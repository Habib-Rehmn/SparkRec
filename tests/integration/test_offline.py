"""Hand-computed protocol checks and a synthetic end-to-end evaluation."""

import json
import math

import pytest
from pyspark.sql import functions as F

from sparkrec.artifacts import write_json
from sparkrec.config import OfflineSettings
from sparkrec.offline.data import baselines, latest, prune, split
from sparkrec.offline.metrics import labels, ranking, rating_metrics
from sparkrec.offline.pipeline import fit, train
from sparkrec.offline.recommend import (
    als_recommendations,
    baseline,
    retrieval_report,
    validate_recommendations,
)

pytestmark = pytest.mark.spark
SCHEMA = "user_id int, movie_id int, rating double, timestamp long"


def test_temporal_boundaries_and_history_local_updates(spark):
    ratings = spark.createDataFrame(
        [
            (1, 1, 2.0, 10),
            (1, 2, 4.0, 19),
            (1, 1, 5.0, 20),
            (2, 2, 3.0, 20),
            (1, 1, 3.0, 30),
            (1, 1, 4.0, 30),
        ],
        SCHEMA,
    )
    settings = OfflineSettings(validation_start=20, test_start=30)
    early, validation, test, report = split(ratings, settings)
    assert report["events"] == {"train": 2, "validation": 2, "test": 2}
    assert latest(early).filter("movie_id = 1").first().rating == 2
    assert latest(early.unionByName(validation)).filter("movie_id = 1").first().rating == 5
    assert latest(test).first().rating == 4
    with pytest.raises(ValueError, match="Empty temporal fold"):
        split(ratings, settings.model_copy(update={"validation_start": 5}))


def test_iterative_training_only_pruning_and_baseline(spark):
    ratings = spark.createDataFrame(
        [
            (1, 1, 5.0, 1),
            (1, 2, 3.0, 1),
            (2, 1, 4.0, 1),
            (2, 2, 2.0, 1),
            (3, 2, 5.0, 1),
            (3, 3, 5.0, 1),
        ],
        SCHEMA,
    )
    settings = OfflineSettings(min_user_ratings=2, min_movie_ratings=2, prior_weight=2)
    filtered, report = prune(ratings, settings)
    assert report["removed_pairs"] == 2
    assert report["iterations"] == 3
    assert {r.user_id for r in filtered.collect()} == {1, 2}
    stats = {r.movie_id: r for r in baselines(filtered, settings).collect()}
    assert stats[1].popularity == 2
    assert stats[1].weighted_rating == 4.0  # (2*4.5 + 2*3.5)/4
    assert stats[2].weighted_rating == 3.0
    with pytest.raises(ValueError, match="removed every"):
        prune(ratings, settings.model_copy(update={"min_user_ratings": 10}))


def test_hand_computed_metrics_short_lists_and_empty_cohort(spark):
    relevant = spark.createDataFrame([(1, 10), (1, 30), (2, 20)], "user_id int, movie_id int")
    all_relevant = relevant.union(spark.createDataFrame([(1, 99), (3, 99)], relevant.schema))
    users = spark.createDataFrame([(1,), (2,), (3,)], "user_id int")
    recs = spark.createDataFrame(
        [(1, 10, 1), (1, 20, 2), (1, 30, 3)], "user_id int, movie_id int, rank int"
    )
    result = ranking(recs, relevant, all_relevant, users, (3,))["3"]
    assert result["users"] == 2
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["map"] == pytest.approx((1 + 2 / 3) / 4)
    assert result["ndcg"] == pytest.approx((1 + 0.5) / (1 + 1 / math.log2(3)) / 2)
    assert result["hit_rate"] == 0.5
    assert result["recall_including_unreachable"] == pytest.approx(2 / 9)
    empty = ranking(recs, relevant, all_relevant, users.filter("user_id = 3"), (3,))["3"]
    assert empty["users"] == 0
    assert empty["ndcg"] is None


def test_positive_label_exclusions(spark):
    heldout = spark.createDataFrame(
        [(1, 1, 5.0, 5), (1, 2, 5.0, 5), (1, 2, 2.0, 6), (2, 3, 4.0, 5), (3, 99, 5.0, 5)], SCHEMA
    )
    seen = spark.createDataFrame([(1, 1)], "user_id int, movie_id int")
    catalog = spark.createDataFrame([(1,), (2,), (3,)], "movie_id int")
    eligible, _, report = labels(heldout, seen, catalog, 4)
    assert eligible.collect()[0].user_id == 2
    assert report["previously_seen_positive_pairs"] == 1
    assert report["unreachable_positive_pairs"] == 1
    assert report["users_without_eligible_positives"] == 2


def test_als_fallback_seen_exclusion_exhaustion_and_rmse(spark):
    training = spark.createDataFrame(
        [(u, m, float((u + m) % 5 + 1), 1) for u in (1, 2) for m in range(1, 5)], SCHEMA
    )
    settings = OfflineSettings(stored_k=20, initial_candidates=1, max_candidates=2, max_iter=2)
    catalog = baselines(training, settings)
    model = fit(training, settings, 2, 0.1)
    users = spark.createDataFrame([(1,), (99,)], "user_id int")
    seen = training.select("user_id", "movie_id")
    recs = als_recommendations(model, users, catalog, seen, settings)
    validate_recommendations(recs, seen, catalog, 20)
    assert recs.filter("user_id = 1").count() == 0
    assert recs.filter("user_id = 99 AND source = 'popularity'").count() == 4
    report = retrieval_report(recs, users, seen, catalog, 20)
    assert report["catalog_exhausted_users"] == 2
    assert report["fallback_users"] == 1
    heldout = spark.createDataFrame([(1, 1, 5.0, 2), (99, 1, 5.0, 2), (1, 99, 5.0, 2)], SCHEMA)
    prediction = rating_metrics(model, heldout)
    assert prediction["scorable"] == 1
    assert prediction["missing_factor_rows"] == 2
    assert prediction["coverage"] == pytest.approx(1 / 3)
    assert math.isfinite(prediction["rmse"])
    bad = baseline(users, catalog, seen, "popularity", 20).withColumn("rank", F.lit(2))
    with pytest.raises(ValueError, match="ranks"):
        validate_recommendations(bad, seen, catalog, 20)


def test_offline_pipeline_lineage_and_selection(spark, config, tmp_path):
    source = tmp_path / "ingestion"
    source.mkdir()
    rows = [
        (u, m, float(4 + (u + m) % 2), 10)
        for u in range(1, 5)
        for m in range(1, 7)
        if (u + m) % 3 != 0
    ]
    rows += [(1, 2, 5.0, 20), (2, 1, 4.0, 20), (8, 3, 5.0, 20)]
    rows += [(1, 5, 5.0, 30), (2, 4, 4.0, 30), (9, 3, 5.0, 30)]
    spark.createDataFrame(rows, SCHEMA).write.parquet(str(source / "ratings"))
    write_json(
        source / "manifest.json",
        {"status": "complete", "input_sha256": {"ratings.csv": "synthetic"}},
    )
    settings = OfflineSettings(
        output_dir=tmp_path / "offline",
        validation_start=20,
        test_start=30,
        min_user_ratings=1,
        min_movie_ratings=1,
        ranks=[2],
        max_iter=2,
        initial_candidates=2,
        max_candidates=6,
        audit_users=1,
    )
    config = config.model_copy(update={"offline": settings})
    result = train(config, spark, source, "synthetic")
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["purpose"] == "offline_evaluation_not_serving"
    assert manifest["input_sha256"]["ratings.csv"] == "synthetic"
    assert len(manifest["input_manifest_sha256"]) == 64
    assert manifest["parquet_sha256"]
    report = json.loads((result / "report.json").read_text())
    assert report["test"]["models"]["als_0"]["all_users"]["10"]["users"] == 3
    assert (result / "selection.json").exists()
    assert (
        spark.read.parquet(str(result / "test/training")).agg(F.max("timestamp")).first()[0] == 20
    )
    with pytest.raises(FileExistsError):
        train(config, spark, source, "synthetic")
