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

See the [five-model student-test comparison](reports/test-evaluation/five-models/README.md) for measured results before and after HMM decoding.

Compare `ok.top1_accuracy` against `ok.nearest_top1_accuracy` for the primary ranking comparison. Top-1 accuracy counts every labelled observation, including candidate misses and observations without candidates. Candidate classification metrics use a probability threshold of 0.5. Direct continuity measures whether adjacent scored observations within a GPS segment choose the same arc or connected arcs. The run ends at independent candidate scoring, before sequential route decoding.

For a quick run on a few trips:

```sh
uv run --locked --extra models python -m pathsnap.models.train --root "../Student folder" \
  --folds 2 --trip-limit-per-partition 3 --out reports/model-training-smoke
```

## Tune logistic regression with Bayesian optimization

Logistic regression led the five-model comparison on mean training-fold top-1 accuracy.
The tuner maximizes the same observation-level metric on `ok` labels, using five
trip-grouped folds and fitting the scaler separately inside each fold. The search
varies `C` from 0.001 to 1,000 on a log scale to cover strong through weak
regularization, and varies the positive class weight from 0.5 to 2 on a log scale
to change how much missed reference arcs cost during fitting. The negative class
weight stays at 1. A Gaussian process models the validation scores, and expected
improvement chooses each new setting after five initial trials, including the
untuned baseline. Ties keep the earliest trial.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --locked --extra models python -m pathsnap.models.tune --root "../Student folder"
```

The default run evaluates 20 settings on `train`. After selection, it refits the
baseline and winner on all training trips and evaluates both on `student_test`.
`reports/model-tuning/` receives the trial and fold results, candidate macro F1
and log loss, fit and inference times, and both `.joblib` artifacts. Use a fresh
`--out` directory for another run. Pass `logistic_regression_tuned.joblib` to the
decoder with `--model` or to the API through `PATHSNAP_MODEL`.

The [recorded run](reports/tuning/logistic_regression.json) used 80 training trips
and 10 student-test trips. Trial 14 selected `C=1000` and a positive class weight
of `2`, both at the upper edge of the search. Untuned and tuned results were,
respectively:

- Mean validation top-1 accuracy was 92.487% and 92.510%. Mean validation
  candidate macro F1 was 94.867% and 94.865%.
- Student-test top-1 accuracy was 94.724% and 94.829%. Candidate macro F1 was
  96.611% and 96.560%, and log loss was 0.06232 and 0.06622.
- Student-test candidate probability scoring took 8.76 ms and 9.01 ms across the
  full partition in this run. These timings exclude feature generation and decoding.

The search took 121 seconds, including fold fits and scoring. The full run took
173 seconds, including data preparation, refits, and student-test evaluation.
The small ranking gain came with slightly worse candidate classification and
probability scores.

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

## Run the FastAPI backend on Laptop 1

On Linux, run `./start-server.sh` from the repository root and enter
`student_test` for rehearsal or `evaluator_test` for the evaluation. The launcher
expects `../Student folder/student_test/` and `../evaluator_test/` alongside the
checkout, and loads `reports/model-tuning/logistic_regression_tuned.joblib` by
default. Set `PATHSNAP_MODEL` to an absolute path to use another fitted logistic
regression artifact.
On Laptop 2, use `http://<laptop-1-lan-ip>:8002` as the UI backend URL.

### Manual startup

From the repository root, point the server at the supplied data directory and a
trusted fitted PathSnap artifact. Use the same model library versions as the
training run. The `models` extra installs the libraries used by the five-model
training command.

```sh
uv sync --locked --extra api --extra models
export PATHSNAP_DATA_ROOT="/path/to/Student folder"
export PATHSNAP_MODEL="/path/to/model.joblib"
uv run --locked --extra api --extra models uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

The active partition defaults to `student_test` for rehearsal. For authorized
evaluation, set `PATHSNAP_DATA_ROOT` to the organizer-provided dataset root and
`PATHSNAP_TRIP_PARTITION` to the name of its evaluator partition directory, then
restart the server. The active partition contains the per-trip `.track`,
`.nodes`, `.arcs`, and `.route` files. Configure Laptop 2's UI adapter with
`http://<laptop-1-lan-ip>:8000` as its backend URL.

```sh
curl --fail-with-body http://localhost:8000/health
curl --fail-with-body http://localhost:8000/trips
curl --fail-with-body http://localhost:8000/predict \
  -H 'Content-Type: application/json' --data '{"trip_id":"00000000"}'
```

Use an ID returned by `/trips`. The API returns predicted GeoJSON geometry, with
longitude before latitude and a separate line for each decoded section. It
derives scoring labels from the reference route on Laptop 1 after decoding and
reports the fraction of `ok` observations matched to the exact directed arc.
Unmatched scorable observations count as incorrect. Inference time covers
classifier scoring and decoding. Total processing time also covers loading,
candidate generation, GeoJSON assembly, and scoring. Both are in milliseconds.

OpenAPI is available at `/docs`. Requests for IDs outside the active partition
return `404`. Invalid requests or trips with no drawable route or scorable
observations return `422`. The server returns `503` when inference is unavailable.
Check the server log and restart after fixing any startup configuration or
artifact errors. Run the backend tests with
`uv run --locked --extra api pytest`.

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
