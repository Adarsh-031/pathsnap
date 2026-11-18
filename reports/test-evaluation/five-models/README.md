# Five-model performance with and without HMM

The comparison used the five original fitted classifiers from the same training run, including the untuned Logistic Regression model. All five scored the same 10 `student_test` trips and 12,281 scorable GPS observations, with 80 disjoint training trips. The runs shared candidate features and decoder settings. The evaluation loaded the reference route after prediction on each trip.

## Student-test results

Each line shows exact directed-arc accuracy and mean ordered route F1, respectively. The values go from classifier-only top-1 to classifier plus HMM.

- **LightGBM** — accuracy 94.72% → 93.58%. Route F1 62.13% → 75.91%.
- **CatBoost** — accuracy 94.95% → 93.49%. Route F1 61.92% → 75.71%.
- **XGBoost** — accuracy 94.76% → 93.71%. Route F1 62.53% → 79.42%.
- **Random Forest** — accuracy 94.79% → 93.62%. Route F1 62.27% → 78.75%.
- **Logistic Regression** — accuracy 94.72% → 93.85%. Route F1 63.19% → 79.89%.

HMM improved ordered route F1 by 13.78 to 16.89 percentage points and reduced exact-arc accuracy by 0.87 to 1.46 points. CatBoost had the highest classifier-only accuracy. Logistic Regression had the highest decoded accuracy and route F1.

[Model-level metrics](by-model.csv) · [50 model-trip results](per-trip.csv) · [Settings and model fingerprints](metadata.json)

## How the metrics were calculated

- **Exact directed-arc accuracy** pools correctly matched observations across trips. The denominator contains every observation with an `ok` derived label, including unmatched observations.
- **Ordered route F1** averages trip-level scores equally. A trip score is twice the longest common subsequence length divided by the combined predicted and reference sequence lengths. The top-1 sequence collapses consecutive repeats within a GPS segment. HMM sections include the decoder's connecting arcs.
- **Direct continuity** pools adjacent matched observations with no flagged GPS gap. The pair counts as directly connected when both use the same arc or the first arc ends at the second arc's start. The metric uses the arcs selected at the observations, so a valid connection through intermediate arcs can still count as an indirect step.
- **Candidate metrics** come from the original training run. The CSV includes mean five-fold training CV top-1 accuracy and candidate macro F1, plus student-test candidate macro F1 and log loss. Candidate macro F1 uses a probability threshold of 0.5. These metrics describe classifier probability outputs before decoding.
- **Timings** are the median per trip for candidate probability scoring alone and for probability scoring plus HMM. They exclude loading, feature generation, top-1 ranking, and reference scoring. XGBoost used CPU inference in this run.

## Reproduce

Use the trusted five `.joblib` artifacts and `metrics.json` from the [five-model training command](../../../README.md#train-and-compare-the-five-candidate-classifiers). From the repository root, set `--model-dir` to the directory that holds that run's files. The committed results used the completed `model-training-gpu` run. The artifact SHA-256 values identify the exact inputs.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked --extra models python reports/test-evaluation/compare_models.py \
  --student-root "../Student folder" --model-dir reports/model-training \
  --out /tmp/pathsnap-five-model-results
```
