"""Unseen recommendations with bounded ALS retrieval and labeled fallback."""

from pyspark.sql import Window
from pyspark.sql import functions as F


def top(frame, k):
    order = Window.partitionBy("user_id").orderBy(F.desc("score"), "movie_id")
    return frame.withColumn("rank", F.row_number().over(order)).filter(F.col("rank") <= k)


def baseline(users, catalog, seen, score, k):
    candidates = users.crossJoin(
        catalog.select("movie_id", F.col(score).cast("double").alias("score"))
    )
    return top(candidates.join(seen, ["user_id", "movie_id"], "left_anti"), k).withColumn(
        "source", F.lit(score)
    )


def als_recommendations(model, users, catalog, seen, settings):
    size = catalog.count()
    cap = min(size, settings.max_candidates)
    pool = min(cap, settings.initial_candidates)
    warm = users.join(
        model.userFactors.select(F.col("id").alias("user_id")), "user_id", "left_semi"
    )
    candidates = None
    pending = warm
    while pool:
        retrieved = (
            model.recommendForUserSubset(pending, pool)
            .select("user_id", F.explode("recommendations").alias("rec"))
            .select(
                "user_id",
                F.col("rec.movie_id").alias("movie_id"),
                F.col("rec.rating").cast("double").alias("score"),
            )
        )
        retrieved = retrieved.filter(
            F.col("score").isNotNull() & ~F.isnan("score") & (F.abs("score") != float("inf"))
        ).join(seen, ["user_id", "movie_id"], "left_anti")
        if candidates is not None:
            retrieved = (
                candidates.drop("rank")
                .unionByName(retrieved)
                .dropDuplicates(["user_id", "movie_id"])
            )
        candidates = top(retrieved, settings.stored_k).checkpoint(eager=True)
        if pool == cap:
            break
        counts = candidates.groupBy("user_id").count()
        pending = (
            warm.join(counts, "user_id", "left")
            .fillna(0, ["count"])
            .filter(F.col("count") < settings.stored_k)
            .select("user_id")
        )
        if pending.isEmpty():
            break
        pool = min(cap, pool * 2)
    if candidates is None:
        raise ValueError("Empty candidate catalog")
    personalized = candidates.withColumn("source", F.lit("als"))
    fallback = baseline(
        users,
        catalog,
        seen.unionByName(candidates.select("user_id", "movie_id")).distinct(),
        "popularity",
        settings.stored_k,
    )
    combined = personalized.unionByName(fallback).withColumn(
        "_group", (F.col("source") != "als").cast("int")
    )
    order = Window.partitionBy("user_id").orderBy("_group", "rank", "movie_id")
    return (
        combined.withColumn("rank", F.row_number().over(order))
        .filter(F.col("rank") <= settings.stored_k)
        .drop("_group")
        .checkpoint(eager=True)
    )


def validate_recommendations(recs, seen, catalog, k):
    if not recs.join(seen, ["user_id", "movie_id"], "inner").isEmpty():
        raise ValueError("Recommendations contain previously rated movies")
    if not recs.join(catalog.select("movie_id"), "movie_id", "left_anti").isEmpty():
        raise ValueError("Recommendations contain movies outside training catalog")
    if not recs.filter(
        F.col("score").isNull() | F.isnan("score") | (F.abs("score") == float("inf"))
    ).isEmpty():
        raise ValueError("Invalid recommendation scores")
    invalid = (
        recs.groupBy("user_id")
        .agg(
            F.count("*").alias("n"),
            F.countDistinct("movie_id").alias("movies"),
            F.countDistinct("rank").alias("ranks"),
            F.min("rank").alias("lo"),
            F.max("rank").alias("hi"),
        )
        .filter(
            (F.col("n") != F.col("movies"))
            | (F.col("n") != F.col("ranks"))
            | (F.col("lo") != 1)
            | (F.col("hi") != F.col("n"))
            | (F.col("n") > k)
        )
    )
    if not invalid.isEmpty():
        raise ValueError("Invalid recommendation uniqueness or ranks")


def retrieval_report(recs, users, seen, catalog, k):
    counts = recs.groupBy("user_id").agg(
        F.count("*").alias("n"), F.sum((F.col("source") != "als").cast("int")).alias("fallback")
    )
    seen_counts = (
        seen.join(catalog.select("movie_id"), "movie_id", "left_semi")
        .groupBy("user_id")
        .count()
        .withColumnRenamed("count", "seen_count")
    )
    size = catalog.count()
    rows = users.join(counts, "user_id", "left").join(seen_counts, "user_id", "left").fillna(0)
    result = (
        rows.agg(
            F.count("*").alias("users"),
            F.sum((F.col("n") < k).cast("int")).alias("short_users"),
            F.sum(
                ((F.col("n") < k) & (F.col("n") == size - F.col("seen_count"))).cast("int")
            ).alias("catalog_exhausted_users"),
            F.sum((F.col("fallback") > 0).cast("int")).alias("fallback_users"),
            F.sum("fallback").alias("fallback_rows"),
            F.sum("n").alias("rows"),
        )
        .first()
        .asDict()
    )
    result["catalog_coverage"] = recs.select("movie_id").distinct().count() / size
    result["fallback_user_fraction"] = (
        result["fallback_users"] / result["users"] if result["users"] else None
    )
    return result


def audit(model, users, catalog, seen, recs, settings):
    sample = (
        users.join(model.userFactors.select(F.col("id").alias("user_id")), "user_id", "left_semi")
        .orderBy("user_id")
        .limit(settings.audit_users)
    )
    n = sample.count()
    if not n:
        return {"users": 0, "mean_exact_top_k_overlap": None}
    pairs = sample.crossJoin(catalog.select("movie_id")).join(
        seen, ["user_id", "movie_id"], "left_anti"
    )
    exact = top(
        model.transform(pairs).withColumnRenamed("prediction", "score"), settings.stored_k
    ).select("user_id", "movie_id")
    totals = exact.groupBy("user_id").count().withColumnRenamed("count", "total")
    hits = (
        exact.join(recs.select("user_id", "movie_id"), ["user_id", "movie_id"])
        .groupBy("user_id")
        .count()
    )
    overlap = (
        totals.join(hits, "user_id", "left")
        .fillna(0)
        .agg(F.avg(F.col("count") / F.col("total")))
        .first()[0]
    )
    return {
        "users": n,
        "users_with_unseen_candidates": totals.count(),
        "mean_exact_top_k_overlap": overlap,
        "selection": "lowest warm user IDs",
    }
