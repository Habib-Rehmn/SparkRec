"""Shared Spark lifecycle for CLI jobs and integration tests."""

from contextlib import contextmanager
from pathlib import Path

from pyspark.sql import SparkSession

from sparkrec.config import SparkSettings


@contextmanager
def spark_session(settings: SparkSettings, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.appName("SparkRec")
        .master(settings.master)
        .config("spark.driver.memory", settings.driver_memory)
        .config("spark.sql.shuffle.partitions", settings.shuffle_partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.csv.parser.columnPruning.enabled", "false")
        .config("spark.sql.warehouse.dir", str(work_dir / "warehouse"))
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(settings.log_level)
    spark.sparkContext.setCheckpointDir(str(work_dir / "checkpoints"))
    try:
        yield spark
    finally:
        spark.stop()
