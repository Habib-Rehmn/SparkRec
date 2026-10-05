# Offline evaluation protocol (M2)

`sparkrec train --config configs/dev.yaml --input-run artifacts/ingestion/<run>`
creates an immutable evaluation run under `artifacts/offline/`. `sparkrec demo-small`
chains ingestion of cached data and this training pipeline. Downloads remain explicit.

## Time and history

Two global Unix-second cutoffs split events into train (`t < validation_start`),
validation (`validation_start <= t < test_start`), and test (`t >= test_start`).
Set both cutoffs in `offline` configuration, or leave both unset to target 80/10/10
with Spark approximate quantiles (relative error 0.001). Actual cutoffs and event
counts are saved; timestamp ties stay together. Empty folds fail the run.

Within each available history, the latest rating per user/movie wins. At identical
timestamps the higher rating wins deterministically. Filtering repeatedly removes
users and movies below their configured interaction minimum until stable. Counts,
means, and popularity use only the resulting training pairs. All available history,
including low ratings and pairs removed from training, remains excluded from recommendations.

The two baselines use positive-rating counts and `(v*R + m*C)/(v+m)` respectively.
They share the retained training catalog with explicit ALS. Scores sort descending,
then movie ID ascending. Validation searches the configured rank/regularization grid
(maximum eight candidates), selecting all-user NDCG@10, with configuration-order ties.
The default is one initial candidate: rank 20, regularization 0.1, ten iterations,
seed 42. `selection.json` is written before fitting the test model. Test fitting
recomputes filtering and statistics on train plus validation only. This is not an
all-history serving model; that belongs to M3.

## Retrieval and cohorts

Only held-out users are scored. ALS retrieves 200 candidates initially, doubles the
pool for short-list warm users, and stops at the smaller of catalog size or 2,000.
Seen movies and nonfinite scores are removed. Remaining slots use popularity;
ALS rows precede fallback rows, without comparing their score scales. Unknown users
receive popularity. Lists can be shorter than the requested 50 only when the unseen
catalog is exhausted. Stored rows include user/movie IDs, rank, score, and source;
the parent run directory identifies the snapshot.

Baseline ranking currently evaluates the user/catalog cross product in Spark. This
is appropriate for MovieLens small; larger runs need bounded baseline candidate
retrieval before claiming scale readiness. ALS retrieval is approximate at the cap.
Each stage audits exact unseen top-50 overlap on the five lowest warm user IDs,
including fallback in the delivered list. This deterministic bounded sample is a
sanity check, not a population estimate or a rank-order equivalence guarantee.

Binary relevance uses the latest held-out rating per pair, threshold 4.0. Previously
rated pairs are excluded from ranking truth, but updates remain in rating RMSE.
Reports count excluded seen positives, unreachable positives outside the retained
catalog, and held-out users without eligible positives. Warm and all-user cohorts
both require at least one eligible positive. Warm users have learned factors;
all-user metrics include cold users receiving fallback.

## Formulas

For each user and requested K (5, 10, 20), let R be eligible relevant movies and H
be hits in the first K delivered recommendations. Average the following per-user
values over the explicitly reported cohort:

- Precision: `H/K`, including short lists.
- Recall: `H/|R|`.
- Average precision: sum precision at each relevant rank, divided by `min(K, |R|)`.
- NDCG: sum `hit/log2(rank+1)`, divided by the ideal sum for `min(K, |R|)` hits.
- Hit rate: 1 if H is nonzero, otherwise 0.

Empty cohorts yield JSON null metrics, not zero. A validation stage with no eligible
users cannot select a model and fails. Additional recall includes unreachable unseen
positives in its denominator. It is reported both over the original eligible cohort
and an extended cohort containing users whose positives are all unreachable, so those
users do not disappear from coverage reporting.

RMSE uses finite ALS predictions only. Its report includes total events, scorable
events, missing-factor rows, and prediction coverage. Scores are not probabilities
or calibrated star ratings. All ranking aggregation uses Spark transformations;
only aggregate summaries and the small configuration search results reach the driver.

## Artifacts and limitations

Runs preserve raw temporal folds, filtered training pairs, catalog baseline scores,
relevance labels, each model, recommendation Parquet, JSON reports, and a Markdown
comparison. The manifest records input-manifest and actual ratings-Parquet SHA-256
hashes, original source hashes, configuration, runtime, cutoffs, selection, status,
and timing. Failed runs are retained and never overwrite previous runs. Every table
passes seen-item, catalog, finite-score, uniqueness, and contiguous-rank checks before
its stage report is accepted.

MovieLens records rating activity, not verified viewing or historical catalog
availability. A training-observed catalog is conservative. Global chronological
splits can have many cold users. Baselines may outperform ALS; report the measured
results. Broader tuning, resource profiling, MLflow, and scale evidence are M4 work.
