# M1 verification — 2026-10-05

The Docker image built successfully with Python 3.11.14, Java 17, and Spark 4.0.1. Lint and formatting checks passed. The Docker test suite passed all 22 tests, including six Spark integration tests and an ALS model save/load check.

## MovieLens small

The official ZIP was downloaded, imported through the CLI, verified on cache reuse, and pinned in `configs/dev.yaml` with this SHA-256:

```text
696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436
```

The CLI ingested the dataset in the development image with Spark `local[2]`. A separate process then loaded the persisted Parquet tables and verified their row counts.

| Check | Result |
|---|---:|
| Input / retained ratings | 100,836 |
| Unique users | 610 |
| Movie metadata rows | 9,742 |
| Movies represented in ratings | 9,724 |
| Rejected ratings / movies | 0 / 0 |
| Duplicate rating events | 0 |
| Conflicting movie IDs | 0 |

Generated outputs for this run are in `artifacts/ingestion/movielens-small/`, including the quality report, complete manifest, and Parquet tables. Dataset files and generated outputs are intentionally excluded from Git.

These counts establish ingestion correctness for this snapshot. They are not recommendation-quality results or a performance benchmark.

## Reproduce

From the repository root with Docker daemon access:

```bash
make setup
make test
make fixture
make demo-small
```

To export JUnit test results and the fixture CLI output from the Docker test stage:

```bash
docker build --target test-results \
  --output type=local,dest=artifacts/verification \
  -f docker/spark.Dockerfile .
```

The synthetic fixture checks invalid rows, duplicates, repeated ratings, and quarantine handling. The real dataset verification is separate from the routine test suite so tests do not depend on downloading MovieLens.

Container execution was verified through Docker build stages in this workspace; the Compose definition was validated. The equivalent Compose run commands above are provided for normal interactive use.
