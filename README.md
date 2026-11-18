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
