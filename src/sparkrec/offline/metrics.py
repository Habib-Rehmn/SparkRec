"""Distributed binary-ranking metrics and explicit-rating prediction coverage."""

from pyspark.sql import Window
from pyspark.sql import functions as F

from sparkrec.offline.data import latest


def labels(heldout, seen, catalog, threshold):
    positives = latest(heldout).filter(F.col("rating") >= threshold).select("user_id", "movie_id")
    unseen = positives.join(seen, ["user_id", "movie_id"], "left_anti")
    eligible = unseen.join(catalog.select("movie_id"), "movie_id", "left_semi")
    report = {
        "heldout_users": heldout.select("user_id").distinct().count(),
        "positive_pairs": positives.count(),
        "previously_seen_positive_pairs": positives.count() - unseen.count(),
        "unreachable_positive_pairs": unseen.count() - eligible.count(),
        "eligible_positive_pairs": eligible.count(),
        "eligible_users": eligible.select("user_id").distinct().count(),
    }
    report["users_without_eligible_positives"] = report["heldout_users"] - report["eligible_users"]
    report["unreachable_fraction"] = (
        report["unreachable_positive_pairs"] / unseen.count() if unseen.count() else None
    )
    return eligible, unseen, report


def ranking(recs, relevant, all_relevant, users, ks=(5, 10, 20)):
    eligible_counts = (
        relevant.join(users, "user_id", "left_semi")
        .groupBy("user_id")
        .count()
        .withColumnRenamed("count", "relevant")
    )
    all_counts = (
        all_relevant.join(users, "user_id", "left_semi")
        .groupBy("user_id")
        .count()
        .withColumnRenamed("count", "all_relevant")
    )
    marked = recs.join(
        relevant.withColumn("hit", F.lit(1)), ["user_id", "movie_id"], "left"
    ).fillna(0, ["hit"])
    cumulative = (
        Window.partitionBy("user_id")
        .orderBy("rank")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    marked = marked.withColumn("hits_so_far", F.sum("hit").over(cumulative))
    report = {}
    for k in ks:
        sums = (
            marked.filter(F.col("rank") <= k)
            .groupBy("user_id")
            .agg(
                F.sum("hit").alias("hits"),
                F.sum(F.col("hit") * F.col("hits_so_far") / F.col("rank")).alias("ap_sum"),
                F.sum(F.col("hit") / F.log2(F.col("rank") + 1)).alias("dcg"),
            )
        )
        per_user = (
            eligible_counts.join(all_counts, "user_id").join(sums, "user_id", "left").fillna(0)
        )
        per_user = per_user.withColumn(
            "ideal",
            F.expr(
                f"aggregate(sequence(1, least({k}, relevant)), cast(0 as double), "
                "(acc, x) -> acc + 1D / log2(x + 1))"
            ),
        )
        result = (
            per_user.agg(
                F.count("*").alias("users"),
                F.avg(F.col("hits") / k).alias("precision"),
                F.avg(F.col("hits") / F.col("relevant")).alias("recall"),
                F.avg(F.col("ap_sum") / F.least(F.lit(k), F.col("relevant"))).alias("map"),
                F.avg(F.col("dcg") / F.col("ideal")).alias("ndcg"),
                F.avg((F.col("hits") > 0).cast("double")).alias("hit_rate"),
                F.avg(F.col("hits") / F.col("all_relevant")).alias(
                    "recall_including_unreachable_same_cohort"
                ),
            )
            .first()
            .asDict()
        )
        # A separate cohort also includes users whose positives are ALL unreachable.
        extended = all_counts.join(sums, "user_id", "left").fillna(0)
        result["all_unseen_positive_users"] = extended.count()
        result["recall_including_unreachable"] = extended.agg(
            F.avg(F.col("hits") / F.col("all_relevant"))
        ).first()[0]
        report[str(k)] = result
    return report


def rating_metrics(model, heldout):
    predictions = model.transform(heldout)
    finite = (
        F.col("prediction").isNotNull()
        & ~F.isnan("prediction")
        & (F.abs("prediction") != float("inf"))
    )
    result = (
        predictions.agg(
            F.count("*").alias("total"),
            F.sum(finite.cast("long")).alias("scorable"),
            F.sqrt(F.avg(F.when(finite, F.pow(F.col("prediction") - F.col("rating"), 2)))).alias(
                "rmse"
            ),
        )
        .first()
        .asDict()
    )
    result["missing_factor_rows"] = result["total"] - result["scorable"]
    result["coverage"] = result["scorable"] / result["total"] if result["total"] else None
    return result
