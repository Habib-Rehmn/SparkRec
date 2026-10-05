"""Versioned validation selection, frozen refit, and one held-out test evaluation."""

import json
import logging
import platform
import re
import time
from pathlib import Path
from uuid import uuid4

from pyspark.ml.recommendation import ALS

from sparkrec.artifacts import sha256_file, utc_now, write_json
from sparkrec.offline.data import baselines, prune, split
from sparkrec.offline.metrics import labels, ranking, rating_metrics
from sparkrec.offline.recommend import (
    als_recommendations,
    audit,
    baseline,
    retrieval_report,
    validate_recommendations,
)

LOG = logging.getLogger(__name__)


def fit(training, settings, rank, reg):
    return ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        rank=rank,
        regParam=reg,
        maxIter=settings.max_iter,
        seed=settings.seed,
        implicitPrefs=False,
        coldStartStrategy="nan",
        numUserBlocks=2,
        numItemBlocks=2,
    ).fit(training.orderBy("user_id", "movie_id"))


def stage(history, heldout, settings, directory, candidates):
    start = time.monotonic()
    directory.mkdir()
    training, filtering = prune(history, settings)
    training.write.parquet(str(directory / "training"))
    catalog = baselines(training, settings).checkpoint(eager=True)
    catalog.write.parquet(str(directory / "baselines"))
    users = heldout.select("user_id").distinct().checkpoint(eager=True)
    seen = history.select("user_id", "movie_id").distinct().checkpoint(eager=True)
    relevant, unseen, coverage = labels(heldout, seen, catalog, settings.relevance_threshold)
    relevant = relevant.checkpoint(eager=True)
    unseen = unseen.checkpoint(eager=True)
    warm = users.join(training.select("user_id").distinct(), "user_id", "left_semi")
    relevant.write.parquet(str(directory / "relevance"))
    reports = {}

    def evaluate(name, recs, model=None):
        recs = recs.checkpoint(eager=True)
        validate_recommendations(recs, seen, catalog, settings.stored_k)
        recs.write.parquet(str(directory / name / "recommendations"))
        report = {
            "all_users": ranking(recs, relevant, unseen, users),
            "warm_users": ranking(recs, relevant, unseen, warm),
            "retrieval": retrieval_report(recs, users, seen, catalog, settings.stored_k),
        }
        if model is not None:
            report["rating_prediction"] = rating_metrics(model, heldout)
            report["exact_retrieval_audit"] = audit(model, users, catalog, seen, recs, settings)
        reports[name] = report
        write_json(directory / name / "report.json", report)
        LOG.info("%s / %s: NDCG@10=%s", directory.name, name, report["all_users"]["10"]["ndcg"])

    for score in ("popularity", "weighted_rating"):
        evaluate(score, baseline(users, catalog, seen, score, settings.stored_k))
    choices = []
    for index, (rank, reg) in enumerate(candidates):
        name = f"als_{index}"
        LOG.info("Fitting %s / %s: rank=%s reg=%s", directory.name, name, rank, reg)
        model = fit(training, settings, rank, reg)
        model.write().save(str(directory / name / "model"))
        evaluate(name, als_recommendations(model, users, catalog, seen, settings), model)
        choices.append(
            {
                "name": name,
                "rank": rank,
                "reg_param": reg,
                "ndcg_at_10": reports[name]["all_users"]["10"]["ndcg"],
            }
        )
    report = {
        "filtering": filtering,
        "labels": coverage,
        "models": reports,
        "candidates": choices,
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(directory / "report.json", report)
    return report


def train(config, spark, input_run: Path, run_id=None):
    input_run = input_run.resolve()
    source_path = input_run / "manifest.json"
    source = json.loads(source_path.read_text())
    if source.get("status") != "complete":
        raise ValueError("Training requires a complete ingestion run")
    run_id = run_id or uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run_id; use 1-80 letters, digits, underscores or hyphens")
    directory = config.offline.output_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "purpose": "offline_evaluation_not_serving",
        "started_at": utc_now(),
        "input_run": str(input_run),
        "input_manifest_sha256": sha256_file(source_path),
        "input_sha256": source.get("input_sha256"),
        "dataset": source.get("dataset"),
        "config": config.model_dump(mode="json"),
        "spark_version": spark.version,
        "python_version": platform.python_version(),
        "execution": {
            "master": spark.sparkContext.master,
            "default_parallelism": spark.sparkContext.defaultParallelism,
        },
        "git_revision": source.get("git_revision"),
    }
    write_json(directory / "manifest.json", manifest)
    try:
        ratings = spark.read.parquet(str(input_run / "ratings"))
        # Fingerprint the actual persisted input files, not only the original CSVs.
        manifest["parquet_sha256"] = {
            str(p.relative_to(input_run)): sha256_file(p)
            for p in sorted((input_run / "ratings").rglob("*.parquet"))
        }
        train_rows, validation, test, cutoffs = split(ratings, config.offline)
        manifest["split"] = cutoffs
        for name, frame in (("train", train_rows), ("validation", validation), ("test", test)):
            frame.write.parquet(str(directory / "folds" / name))
        write_json(directory / "manifest.json", manifest)
        grid = [(rank, reg) for rank in config.offline.ranks for reg in config.offline.reg_params]
        validation_report = stage(
            train_rows, validation, config.offline, directory / "validation", grid
        )
        eligible = [c for c in validation_report["candidates"] if c["ndcg_at_10"] is not None]
        if not eligible:
            raise ValueError("No eligible validation cohort; cannot select an ALS configuration")
        selected = max(eligible, key=lambda c: c["ndcg_at_10"])
        # This immutable file is written BEFORE test model fitting or test metrics.
        write_json(
            directory / "selection.json",
            {
                "metric": "all_users.ndcg@10",
                "tie_break": "configuration order",
                "selected": selected,
                "frozen_at": utc_now(),
            },
        )
        manifest["selected"] = selected
        write_json(directory / "manifest.json", manifest)
        test_report = stage(
            train_rows.unionByName(validation),
            test,
            config.offline,
            directory / "test",
            [(selected["rank"], selected["reg_param"])],
        )
        report = {
            "split": cutoffs,
            "selected": selected,
            "validation": validation_report,
            "test": test_report,
        }
        write_json(directory / "report.json", report)
        lines = [
            "# Offline evaluation",
            "",
            "Selection: validation all-user NDCG@10. Test refit uses train + validation only.",
            "",
            "| Stage | Model | Users | NDCG@10 | Recall@10 |",
            "|---|---|---:|---:|---:|",
        ]
        for label, result in (("validation", validation_report), ("test", test_report)):
            for name, model_report in result["models"].items():
                metric = model_report["all_users"]["10"]
                lines.append(
                    f"| {label} | {name} | {metric['users']} | "
                    f"{metric['ndcg']} | {metric['recall']} |"
                )
        (directory / "report.md").write_text("\n".join(lines) + "\n")
        manifest["status"] = "complete"
        return directory
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.update(finished_at=utc_now(), elapsed_seconds=time.monotonic() - start)
        write_json(directory / "manifest.json", manifest)
