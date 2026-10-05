# SparkRec

Movie recommendation pipelines built with PySpark. The current milestone implements **M1: configuration, dataset acquisition, quality validation, and Parquet ingestion**. The roadmap is in [project.md](project.md); recommendation training, evaluation, API, and dashboard follow in later milestones.

## Run with Docker

Prerequisites: Docker Engine with Compose and access to its daemon. Start with roughly 4 GB available to the container and enough disk for the Spark image, dependency/build caches, and generated data. The default job uses two CPU cores.

```bash
make setup
make test
make fixture
make demo-small
```

`make test` builds a test stage that runs lint, unit tests, Spark integration tests, and a tiny ALS save/load check. Docker may reuse a successful cached test layer when its inputs have not changed. `make demo-small` is the M1 data demo: it downloads MovieLens small, validates it, and writes Parquet. It does not yet generate movie recommendations.

Builds use a pinned Python 3.11 image, Java 17, Spark 4.0.1, and a frozen Python dependency lock. Downloads happen explicitly at runtime. The development container runs as your user when invoked through Make, so generated host files retain usable ownership.

The initial build downloads Spark's substantial distribution and may take several minutes. Later builds reuse Docker layers and the dependency cache. PySpark's version-specific dependency metadata is copied from PyPI in `pyproject.toml` to avoid downloading that archive just to resolve the lock; refresh the metadata when upgrading Spark. See the [runtime decision](docs/architecture/adr-001-development-runtime.md).

Optional settings are listed in [.env.example](.env.example). Configuration paths are relative to their YAML file, independent of the shell's working directory. The `configs` directory is mounted read-only; changing configuration does not require rebuilding. Source changes do require `make build`.

## Commands

```bash
# Resolve and validate configuration without starting Spark.
docker compose run --rm pipeline validate-config --config configs/dev.yaml

# Check Java, Spark, and ALS.
make smoke

# Use a ZIP downloaded separately (place it under data/ on the host).
docker compose run --rm pipeline download --config configs/dev.yaml \
  --archive /app/data/ml-latest-small.zip

# Choose a unique run name; existing runs are never overwritten.
docker compose run --rm pipeline ingest --config configs/dev.yaml --run-id first-run
```

MovieLens inputs and their original README/license are stored under `data/raw/ml-latest-small/`. A ZIP checksum and per-file checksums are recorded in `dataset.json`; cached data is verified on reuse and before ingestion. The development configuration pins `dataset.archive_sha256` to the verified small-dataset release; use this field to enforce a particular archive. A changed or corrupt cache fails instead of being silently replaced. To intentionally acquire a different dataset snapshot, use a new input directory and update the timestamp/checksum configuration.

The development configuration uses a September 2018 snapshot cutoff matching the currently targeted small release. A changed upstream dataset may need a new cutoff; future timestamps are not silently accepted.

## Outputs and quality rules

Each ingestion creates a unique directory in `artifacts/ingestion/` (or `artifacts/fixture/` for the fixture):

```text
<run_id>/
  manifest.json
  quality_report.json
  ratings/                 # Parquet: user_id, movie_id, rating, timestamp
  movies/                  # Parquet: movie_id, title, genres
  quarantine/ratings/      # Original fields, raw record, and reason codes
  quarantine/movies/
```

The manifest records status, resolved configuration, source hashes, runtime versions, timings, and Git revision when available. Failed quality checks retain reports and quarantined rows; only complete runs should be used downstream. Spark scratch/checkpoint files live in the separate `_spark/` directory.

Validation rejects malformed records, missing/invalid IDs, invalid ratings, invalid timestamps, and unknown movie references. Contradictory metadata for one movie ID fails the run. Unknown genres become an empty array. Exact normalized duplicate events are removed, while repeated ratings at different timestamps are retained for later temporal splitting. Same-timestamp rating conflicts are counted and retained for fold-specific handling in M2.

The normal development profile accepts zero rejected rows. The synthetic fixture deliberately contains seven invalid ratings and one duplicate; its threshold allows these so quarantine handling can be demonstrated. One rejected record can have multiple reasons, so reason counts may exceed the rejected-row count.

MovieLens's one-record-per-line CSV files are parsed partition by partition with Python's strict CSV reader, using an explicit Spark schema. This preserves wrong-width and malformed-quote records instead of silently truncating them. All subsequent type validation, joins, deduplication, and aggregation use Spark DataFrames. Multiline CSV records are outside this input contract.

## Local development

Use Python 3.11 or 3.12, Java 17+, and `uv`. Select Java 17 explicitly if the default Java installation is older. Spark needs permission to create local network sockets even in local mode.

```bash
uv sync --frozen
uv run ruff check src tests
uv run pytest -q -m 'not spark'
uv run pytest -q
uv run sparkrec ingest --config configs/fixture.yaml
```

Set `JAVA_HOME` to your Java 17 installation if needed. For a host with hostname-resolution problems, set `SPARK_LOCAL_IP=127.0.0.1` for local-mode tests. The Docker image already supplies both settings.

The test suite uses synthetic data and requires no dataset download. The real MovieLens run is a separate verification step. Never treat example or fixture counts as benchmark results.

## Data attribution

MovieLens is provided by GroupLens Research. Download links and usage terms are available from [MovieLens Latest](https://grouplens.org/datasets/movielens/latest/) and the [small dataset README](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). Preserve the downloaded README and follow its terms for any redistribution or publications.

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19. [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

## Cleanup

`docker compose down` stops this project's services. It does not delete the bind-mounted `data/` or `artifacts/` directories. Remove specific generated runs or datasets manually only when they are no longer needed; normal commands never delete previous runs.
