# M2 verification — 2026-10-05

M2 completed on the existing verified MovieLens small ingestion run using Python 3.11.14, Spark 4.0.1, Java 17, and local[2] in Docker. Ruff lint and formatting passed. All **35 tests passed**, including 12 Spark integration tests. JUnit results are in `artifacts/verification-m2/tests.xml`.

The suite covers temporal boundaries and ties, history-local rating updates, iterative training-only filtering, hand-calculated metrics, empty cohorts, unreachable positives, fallback, exhausted catalogs, RMSE coverage, recommendation invariants, model persistence, and end-to-end artifact lineage.

## Measured results

The source contains 100,836 ratings. The saved temporal folds contain 80,574 training, 10,092 validation, and 10,170 test events. Cutoffs are Unix seconds `1457846226` and `1496209411`. The default single ALS candidate uses rank 20, regularization 0.1, ten iterations, and seed 42. Selection was frozen before the test refit.

| Stage | Model | Eligible users | NDCG@10 | Recall@10 |
|---|---|---:|---:|---:|
| validation | popularity | 57 | 0.3513 | 0.0836 |
| validation | weighted_rating | 57 | 0.2675 | 0.0595 |
| validation | als_0 | 57 | 0.3382 | 0.0692 |
| test | popularity | 74 | 0.2912 | 0.0810 |
| test | weighted_rating | 74 | 0.2144 | 0.0537 |
| test | als_0 | 74 | 0.2732 | 0.0721 |

These are fallback-enabled, all-user metrics on users with eligible positive labels. Popularity outperformed this initial ALS configuration on both validation and test. This is a first measured baseline, not evidence that ALS improves recommendation quality.

Test RMSE was 0.8702 on 994 of 10,170 events (**9.77% prediction coverage**). The remaining 9,176 events lacked learned factors. The test stage served 76 users; 49 used popularity fallback. All received 50 distinct unseen recommendations. Only 74 users had eligible positive labels for the main ranking cohort. Of 4,165 positive test pairs, 1,015 (24.37%) were outside the training catalog. Additional recall including unreachable positives is recorded in JSON.

The bounded retrieval audit matched exact unseen top-50 membership for all five sampled warm users in each stage. This sample does not establish exact retrieval for every user. Baseline generation currently uses a distributed user/catalog cross product; larger-data optimization and benchmarking remain M4 work.

## Artifacts and reproduction

The complete run is in `artifacts/offline/movielens-small-m2/`: manifest, frozen selection, raw temporal folds, filtered training tables, saved validation/test ALS models, baseline scores, relevance tables, recommendations, JSON reports, and `report.md`. Inputs and generated artifacts remain excluded from version control.

```bash
make build
make test
make train INPUT_RUN=artifacts/ingestion/movielens-small
# From downloaded raw data, ingest and train in one command:
make demo-small
```

Execution in this workspace used Docker build stages, including the real-data CLI run. Compose commands are provided for normal interactive use; a Compose training run was not separately executed here. Runtime was about 76 seconds for this small-data run, while tests were also running; this is not a controlled performance benchmark.

The protocol is documented in [evaluation.md](evaluation.md). M3 remains separate: all-history serving fit, SQLite publication, FastAPI, and the user-selected Next.js/TypeScript frontend.
