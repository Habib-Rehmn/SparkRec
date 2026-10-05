"""Distributed CSV parsing and data-quality transformations."""

from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from sparkrec.ingestion.csv_records import check_header, parse_partition

RATING_COLUMNS = ("userId", "movieId", "rating", "timestamp")
MOVIE_COLUMNS = ("movieId", "title", "genres")


def read_raw(spark: SparkSession, path: Path, columns: tuple[str, ...]) -> DataFrame:
    check_header(path, columns)
    schema = StructType(
        [StructField(f"raw_{name}", StringType()) for name in columns]
        + [StructField("raw_record", StringType()), StructField("parse_error", StringType())]
    )
    # Partition-local parsing detects wrong field counts and quote errors that
    # permissive CSV readers can otherwise lose. Validation below stays in SQL.
    records = spark.sparkContext.textFile(str(path)).mapPartitionsWithIndex(
        lambda index, lines: parse_partition(index, lines, len(columns))
    )
    return spark.createDataFrame(records, schema)


def _reasons(*checks: tuple[Column, str]) -> Column:
    return F.array_compact(F.array(*[F.when(condition, reason) for condition, reason in checks]))


def _missing(name: str) -> Column:
    return F.col(name).isNull() | (F.trim(F.col(name)) == "")


def validate_movies(raw: DataFrame) -> DataFrame:
    typed = raw.withColumn("movie_id", F.expr("try_cast(raw_movieId as int)")).withColumn(
        "title", F.trim("raw_title")
    )
    genres = F.array_sort(F.array_distinct(F.split(F.col("raw_genres"), r"\|")))
    return typed.withColumn(
        "genres",
        F.when(
            _missing("raw_genres") | (F.col("raw_genres") == "(no genres listed)"),
            F.array().cast("array<string>"),
        ).otherwise(genres),
    ).withColumn(
        "reasons",
        _reasons(
            (F.col("parse_error").isNotNull(), "malformed_csv"),
            (_missing("raw_movieId"), "missing_movie_id"),
            (F.col("movie_id").isNull() | (F.col("movie_id") <= 0), "invalid_movie_id"),
            (_missing("raw_title"), "missing_title"),
        ),
    )


def validate_ratings(raw: DataFrame, movies: DataFrame, snapshot_epoch: int) -> DataFrame:
    typed = raw.select(
        "*",
        F.expr("try_cast(raw_userId as int)").alias("user_id"),
        F.expr("try_cast(raw_movieId as int)").alias("movie_id"),
        F.expr("try_cast(raw_rating as double)").alias("rating"),
        F.expr("try_cast(raw_timestamp as bigint)").alias("timestamp"),
    )
    known = movies.select("movie_id").distinct().withColumn("known_movie", F.lit(True))
    joined = typed.join(known, "movie_id", "left")
    rating = F.col("rating")
    timestamp = F.col("timestamp")
    return joined.withColumn(
        "reasons",
        _reasons(
            (F.col("parse_error").isNotNull(), "malformed_csv"),
            (_missing("raw_userId"), "missing_user_id"),
            (F.col("user_id").isNull() | (F.col("user_id") <= 0), "invalid_user_id"),
            (_missing("raw_movieId"), "missing_movie_id"),
            (F.col("movie_id").isNull() | (F.col("movie_id") <= 0), "invalid_movie_id"),
            (_missing("raw_rating"), "missing_rating"),
            (
                rating.isNull()
                | F.isnan(rating)
                | (rating < 0.5)
                | (rating > 5.0)
                | (F.pmod(rating * 2, F.lit(1)) != 0),
                "invalid_rating",
            ),
            (_missing("raw_timestamp"), "missing_timestamp"),
            (
                timestamp.isNull() | (timestamp <= 0) | (timestamp > snapshot_epoch),
                "invalid_timestamp",
            ),
            (
                F.col("movie_id").isNotNull()
                & (F.col("movie_id") > 0)
                & F.col("known_movie").isNull(),
                "unknown_movie",
            ),
        ),
    ).drop("known_movie")


def quality_summary(frame: DataFrame) -> dict:
    counts = frame.agg(
        F.count("*").alias("total"),
        F.count(F.when(F.size("reasons") > 0, 1)).alias("rejected"),
    ).first()
    reasons = frame.select(F.explode("reasons").alias("reason")).groupBy("reason").count()
    return {
        "total": counts.total,
        "rejected": counts.rejected,
        "accepted_before_deduplication": counts.total - counts.rejected,
        "rejected_fraction": counts.rejected / counts.total if counts.total else 0.0,
        "reason_counts": {row.reason: row["count"] for row in reasons.collect()},
    }
