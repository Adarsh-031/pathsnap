# PathSnap

PathSnap matches GPS observations to directed road arcs. The [data pipeline guide](docs/data_pipeline.md) describes the supplied dataset and how the candidate features and observation labels are derived.

## Train and compare the five candidate classifiers

With Python 3.12, uv, and the supplied `Student folder` available, run from the repository root:

```sh
uv sync --locked --extra models
uv run --locked --extra models python -m pathsnap.models.train --root "../Student folder"
```

The command fits LightGBM, CatBoost, XGBoost, Random Forest, and Logistic Regression on the same candidate features. It uses five trip-grouped folds on `train` and fits on candidates from observations with `ok` derived labels. Logistic Regression scales its inputs inside each fit. Each model is then refit on the full training partition and scored on `student_test`.

The default output directory is `reports/model-training/`. It contains the five fitted estimators and `metrics.json` with model settings, package versions, trip assignments, fit and inference times, and metrics for each validation fold and the student-test partition. Use `--out` to retain separate runs.

Compare `ok.top1_accuracy` against `ok.nearest_top1_accuracy` for the primary ranking comparison. Top-1 accuracy counts every labelled observation, including candidate misses and observations without candidates. Candidate classification metrics use a probability threshold of 0.5. Direct continuity measures whether adjacent scored observations within a GPS segment choose the same arc or connected arcs. The run ends at independent candidate scoring, before sequential route decoding.

For a quick run on a few trips:

```sh
uv run --locked --extra models python -m pathsnap.models.train --root "../Student folder" \
  --folds 2 --trip-limit-per-partition 3 --out reports/model-training-smoke
```

## Reconstruct a route

Build GPS candidates and decode a directed route with the HMM. The command writes
JSON with one match per observation and an ordered list of continuous sections.

```bash
uv run python -m pathsnap.reconstruction.run_trip --root "/path/to/Student folder" --trip 00000001
```

Pass `--model /path/to/model.joblib` to use a trusted PathSnap training artifact.
The artifact supplies the classifier and candidate-generation settings. Include
`--extra models` after `uv run` to load artifacts from the five-model training
command. Without a model, the decoder uses GPS-to-road distance for emissions.

In Python, call `pathsnap.reconstruction.reconstruct_route(trip, classifier=estimator)`
with a fitted binary classifier. For existing candidate scores, pass the observation
table, candidate table, and positive-class probabilities in candidate row order
to `pathsnap.reconstruction.decode_route` along with the trip's `RoadGraph`.

Each match gives the observation index, selected directed arc ID, and section index.
An observation without candidates has null arc and section IDs. Each section lists
the traversed arc IDs, including connecting arcs between observations. The inference
command uses the GPS track and road network.

## Decoder

Each hidden state is a candidate directed arc and the observation's projected
position on it. Emission scores are the log of the classifier probability, or
a Gaussian log score based on perpendicular distance with a 10 m standard
deviation. Viterbi adds the emission score to the best predecessor's cumulative
score plus the transition score, then traces back the best path in each section.

Transitions use the shortest directed distance between the projected positions,
including partial arcs. The log transition score is the negative absolute
difference between that distance and the observed GPS displacement, scaled by
20 m. The decoder discards routes that exceed the GPS displacement by more than
100 m. It allows up to 10 m of backward GPS jitter on the same arc. A larger
backward step requires a directed path back onto that arc.

An observation without candidates, a flagged GPS gap (over 5 seconds by default),
or a step with no reachable candidate from the active Viterbi states ends the
current section. Decoding restarts at the next observation with candidates.

## Development comparison

The comparison used the existing logistic regression artifact (selected by mean
training-fold top-1 accuracy), identical candidate probabilities for both methods,
and default decoder settings on all ten `student_test` trips.

- Pointwise accuracy on 12,281 `ok` derived labels was 94.72% for classifier
  top-1 and 93.85% after Viterbi decoding.
- Directed continuity was 92.11% for classifier top-1 and 99.83% after
  decoding, measured over 19,106 adjacent observations with candidates and no
  flagged gap. Both methods used the same directed transition constraints.
- Mean ordered arc-sequence F1 rose from 63.19% to 79.89%. For each trip,
  F1 was twice the longest common subsequence length divided by the combined
  predicted and reference route lengths. The classifier route collapses consecutive
  repeats within each uninterrupted run. The HMM route includes inferred connecting
  arcs and preserves repeat traversals.

The largest pointwise loss was on trip `00000077`, where Viterbi matched 80
fewer of the 2,222 scorable observations while improving ordered arc-sequence F1
from 26.20% to 42.17%.
