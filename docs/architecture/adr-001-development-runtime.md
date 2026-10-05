# ADR-001: Docker and local Spark for the first milestone

Status: implemented; cluster execution remains a later milestone.

## Context

SparkRec needs a reproducible environment for CSV validation, Parquet output, and subsequent ALS training. Development starts on a single workstation with Docker. The first pipeline must be testable without downloading the full dataset or managing a persistent Spark cluster.

## Decision

Use a digest-pinned Python 3.11 image, Java 17, Spark/PySpark 4.0.1, and a frozen `uv.lock`. Run development Spark with two local threads in a one-shot container. Keep configuration and generated outputs on documented bind mounts and run as a non-root user.

Provide a Docker test stage so integration checks use the same Java/Python environment as the pipeline. Unit tests use synthetic archives and records; Spark integration tests include Parquet roundtrips and an ALS model save/load check.

## Trade-offs

The Spark distribution makes the initial build substantial; Docker layers and a dependency cache avoid repeating downloads during code changes. Version-specific upstream PySpark dependency metadata in `pyproject.toml` avoids downloading its large source archive merely to resolve dependencies. Refresh that metadata when changing the Spark version.

Local Spark does not establish distributed-worker behavior or large-data performance. Add a standalone cluster profile and measure real workloads in M4. The OS package repository can supply updated Java 17 patches on a fresh build; record runtime versions rather than claiming bit-for-bit OS reproducibility.

## CSV handling

Use partition-local strict CSV parsing with an explicit schema for MovieLens's single-line records, then Spark SQL for validation, joining, deduplication, and aggregation. This retains malformed records and wrong field counts for audit. It accepts a small Python parsing cost at the input boundary; benchmark that cost when scaling. Multiline records require a different input adapter.

Rating updates remain intact at ingestion. Deduplicating to the latest user/movie pair happens within the available history of each temporal fold in M2, preventing future events from rewriting earlier training data.
