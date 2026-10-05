"""History-local deduplication, temporal folds, and iterative training pruning."""

from pyspark.sql import Window
from pyspark.sql import functions as F


def latest(frame):
    # A higher rating wins conflicts at the exact same timestamp.
    order = Window.partitionBy("user_id", "movie_id").orderBy(F.desc("timestamp"), F.desc("rating"))
    return frame.withColumn("_row", F.row_number().over(order)).filter("_row = 1").drop("_row")


def split(ratings, settings):
    cuts = [settings.validation_start, settings.test_start]
    if cuts[0] is None:
        cuts = [int(x) for x in ratings.approxQuantile("timestamp", [0.8, 0.9], 0.001)]
    if len(cuts) != 2 or cuts[0] >= cuts[1]:
        raise ValueError("Cannot form distinct temporal cutoffs; supply explicit cutoffs")
    train = ratings.filter(F.col("timestamp") < cuts[0])
    validation = ratings.filter((F.col("timestamp") >= cuts[0]) & (F.col("timestamp") < cuts[1]))
    test = ratings.filter(F.col("timestamp") >= cuts[1])
    counts = dict(
        zip(
            ("train", "validation", "test"),
            (x.count() for x in (train, validation, test)),
            strict=True,
        )
    )
    if not all(counts.values()):
        raise ValueError(f"Empty temporal fold: {counts}")
    return (
        train,
        validation,
        test,
        {"validation_start": cuts[0], "test_start": cuts[1], "events": counts},
    )


def prune(history, settings):
    frame = latest(history).checkpoint(eager=True)
    before = frame.count()
    users_before = frame.select("user_id").distinct().count()
    movies_before = frame.select("movie_id").distinct().count()
    count = before
    iterations = 0
    while True:
        users = frame.groupBy("user_id").count().filter(F.col("count") >= settings.min_user_ratings)
        movies = (
            frame.groupBy("movie_id").count().filter(F.col("count") >= settings.min_movie_ratings)
        )
        narrowed = (
            frame.join(users.select("user_id"), "user_id", "left_semi")
            .join(movies.select("movie_id"), "movie_id", "left_semi")
            .checkpoint(eager=True)
        )
        remaining = narrowed.count()
        iterations += 1
        frame = narrowed
        if remaining == 0:
            raise ValueError("Training-only filtering removed every rating; lower minimum counts")
        if remaining == count:
            break
        count = remaining
    return frame, {
        "deduplicated_pairs": before,
        "retained_pairs": remaining,
        "removed_pairs": before - remaining,
        "iterations": iterations,
        "retained_users": frame.select("user_id").distinct().count(),
        "retained_movies": frame.select("movie_id").distinct().count(),
        "removed_users": users_before - frame.select("user_id").distinct().count(),
        "removed_movies": movies_before - frame.select("movie_id").distinct().count(),
    }


def baselines(training, settings):
    mean = training.agg(F.avg("rating")).first()[0]
    return (
        training.groupBy("movie_id")
        .agg(
            F.count("*").alias("v"),
            F.avg("rating").alias("mean_rating"),
            F.sum((F.col("rating") >= settings.relevance_threshold).cast("long")).alias(
                "popularity"
            ),
        )
        .withColumn(
            "weighted_rating",
            (F.col("v") * F.col("mean_rating") + settings.prior_weight * mean)
            / (F.col("v") + settings.prior_weight),
        )
    )
