"""Small command-line interface with lazy Spark imports."""

import argparse
import json
import logging
from pathlib import Path

from sparkrec import __version__
from sparkrec.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sparkrec", description="SparkRec data pipeline")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", help="Include exception tracebacks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-config", "Validate and print resolved configuration"),
        ("download", "Download and verify a MovieLens dataset"),
        ("ingest", "Validate CSV data and write a versioned Parquet run"),
        ("smoke", "Check the Java/Spark runtime with a tiny ALS fit"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--config", type=Path, default=Path("configs/dev.yaml"))
        if name == "download":
            sub.add_argument("--archive", type=Path, help="Use an already-downloaded MovieLens ZIP")
        if name == "ingest":
            sub.add_argument(
                "--run-id", help="Unique output name (existing runs are never overwritten)"
            )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    try:
        config = load_config(args.config)
        if args.command == "validate-config":
            print(config.model_dump_json(indent=2))
        elif args.command == "download":
            from sparkrec.ingestion.download import download_dataset

            print(json.dumps(download_dataset(config.dataset, args.archive), indent=2))
        else:
            from sparkrec.spark import spark_session

            with spark_session(config.spark, config.output_dir / "_spark") as spark:
                if args.command == "ingest":
                    from sparkrec.ingestion.pipeline import ingest

                    run_dir = ingest(config, spark, args.run_id)
                    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}))
                else:
                    from pyspark.ml.recommendation import ALS

                    rows = [
                        (u, m, float((u + m) % 5 + 1)) for u in range(1, 5) for m in range(1, 5)
                    ]
                    frame = spark.createDataFrame(rows, "user int, item int, rating double")
                    model = ALS(rank=2, maxIter=2, seed=42).fit(frame)
                    if model.recommendForAllUsers(2).count() != 4:
                        raise RuntimeError("ALS smoke check returned an unexpected user count")
                    print(json.dumps({"status": "ok", "spark": spark.version, "als_users": 4}))
        return 0
    except KeyboardInterrupt:
        logging.error("SparkRec interrupted")
        return 130
    except Exception as exc:
        logging.error("SparkRec failed: %s", exc, exc_info=args.debug)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
