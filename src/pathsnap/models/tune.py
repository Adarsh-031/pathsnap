"""Bayesian hyperparameter optimization for PathSnap's logistic regression."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from pathsnap.data.labels import LABEL_SOURCE, LabelConfig
from pathsnap.data.reader import discover_trips
from pathsnap.data.splits import trip_group_folds
from pathsnap.features.candidates import CandidateConfig, FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION

from .artifact import save_model_artifact
from .estimators import logistic_regression
from .train import fit_model, model_parameters, prepare_partition, score_model

MIN_LOG_C, MAX_LOG_C = -3.0, 3.0
MIN_LOG_WEIGHT, MAX_LOG_WEIGHT = -1.0, 1.0
GRID_SIZE = 41
MID = GRID_SIZE // 2
LAST = GRID_SIZE - 1
# Sample the default first, then vary each parameter by itself before fitting the surrogate.
INITIAL_GRID_INDICES = [(MID, MID), (0, MID), (LAST, MID), (MID, 0), (MID, LAST)]
DEFAULT_TRIALS = 20
LENGTH_SCALE = 0.25
SURROGATE_ALPHA = 0.001


def suggest_point(grid: np.ndarray, tried: list[int], scores: list[float]) -> int:
    if len(tried) < len(INITIAL_GRID_INDICES):
        c_index, weight_index = INITIAL_GRID_INDICES[len(tried)]
        return c_index * GRID_SIZE + weight_index

    surrogate = GaussianProcessRegressor(
        kernel=Matern(length_scale=LENGTH_SCALE, nu=2.5), alpha=SURROGATE_ALPHA,
        normalize_y=True, optimizer=None,
    )
    surrogate.fit(grid[tried], scores)
    expected_score, std = surrogate.predict(grid, return_std=True)
    gain = expected_score - max(scores)
    standardized_gain = gain / np.maximum(std, np.finfo(float).eps)
    expected_improvement = gain * norm.cdf(standardized_gain) + std * norm.pdf(standardized_gain)
    expected_improvement[tried] = -np.inf
    return int(np.argmax(expected_improvement))


def trial_parameters(point: np.ndarray) -> dict:
    c, weight = map(float, point)
    return {
        "C": 10 ** (MIN_LOG_C + c * (MAX_LOG_C - MIN_LOG_C)),
        "positive_class_weight": 2 ** (MIN_LOG_WEIGHT + weight * (MAX_LOG_WEIGHT - MIN_LOG_WEIGHT)),
    }


def build_estimator(seed: int, parameters: dict):
    estimator = logistic_regression(seed)
    weight = parameters["positive_class_weight"]
    return estimator.set_params(
        logisticregression__C=parameters["C"],
        logisticregression__class_weight=None if weight == 1.0 else {0: 1.0, 1: weight},
    )


def fold_metrics(metrics: dict) -> dict:
    classification = metrics["ok"]["classification"]
    if classification is None:
        raise ValueError("Validation requires candidates with ok labels in every fold.")
    return {
        "top1_accuracy": metrics["ok"]["top1_accuracy"],
        "macro_f1": classification["macro_f1"],
        "log_loss": classification["log_loss"],
        "inference_seconds": metrics["inference_seconds"],
    }


def run(root: Path, out: Path, folds: int, seed: int, n_trials: int) -> None:
    if out.exists():
        raise ValueError(f"Output path {out} already exists. Choose a new --out path.")
    started = perf_counter()
    config = CandidateConfig()
    train_trips = discover_trips(root, partition="train")
    if len(train_trips) < folds:
        raise ValueError(f"Expected at least {folds} train trips under {root}.")

    obs_train, rows_train = prepare_partition(train_trips, list(train_trips), config)
    splits, fold_data = [], []
    for fit_idx, val_idx in trip_group_folds(obs_train, n_splits=folds, seed=seed):
        validation = obs_train.iloc[val_idx]
        fit_ids = sorted(obs_train.iloc[fit_idx]["trip_id"].unique())
        validation_ids = sorted(validation["trip_id"].unique())
        splits.append({"fit_trip_ids": fit_ids, "validation_trip_ids": validation_ids})
        fold_data.append((
            rows_train[rows_train["trip_id"].isin(fit_ids)],
            rows_train[rows_train["trip_id"].isin(validation_ids)],
            validation,
        ))
    train_preparation_seconds = perf_counter() - started

    search_started = perf_counter()
    axis = np.linspace(0.0, 1.0, GRID_SIZE)
    grid = np.array([(c, weight) for c in axis for weight in axis])
    tried, scores, trials = [], [], []
    for number in range(n_trials):
        trial_started = perf_counter()
        index = suggest_point(grid, tried, scores)
        parameters = trial_parameters(grid[index])
        results = []
        for fit_rows, validation_rows, validation in fold_data:
            estimator = build_estimator(seed, parameters)
            fit_seconds = fit_model(estimator, fit_rows)
            metrics = score_model(estimator, validation_rows, validation)
            results.append({"fit_seconds": fit_seconds, **fold_metrics(metrics)})
        summary = {key: mean(fold[key] for fold in results) for key in results[0]}
        score = summary["top1_accuracy"]
        tried.append(index)
        scores.append(score)
        trials.append({
            "number": number,
            "phase": "initial" if number < len(INITIAL_GRID_INDICES) else "bayesian",
            "parameters": parameters,
            "mean_validation": summary,
            "folds": results,
            "wall_seconds": perf_counter() - trial_started,
        })
        print(f"Trial {number + 1}/{n_trials} {parameters} mean top-1={score:.6f}", flush=True)
    best = max(trials, key=lambda trial: trial["mean_validation"]["top1_accuracy"])
    search_seconds = perf_counter() - search_started

    # The search and tie-breaking are complete before reading any held-out trips.
    preparation_started = perf_counter()
    student_trips = discover_trips(root, partition="student_test")
    if not student_trips:
        raise ValueError(f"Expected student_test trips under {root}.")
    obs_student, rows_student = prepare_partition(student_trips, list(student_trips), config)
    student_preparation_seconds = perf_counter() - preparation_started

    out.mkdir(parents=True)
    comparison = {}
    for name, trial in (("untuned", trials[0]), ("tuned", best)):
        estimator = build_estimator(seed, trial["parameters"])
        fit_seconds = fit_model(estimator, rows_train)
        metrics = score_model(estimator, rows_student, obs_student)
        filename = f"logistic_regression_{name}.joblib"
        save_model_artifact(out / filename, estimator, config)
        comparison[name] = {
            "trial": trial["number"],
            "parameters": model_parameters(estimator),
            "refit_seconds": fit_seconds,
            "student_test": metrics,
            "artifact": filename,
        }

    report = {
        "model": "logistic_regression",
        "objective": "maximize mean validation top-1 accuracy on ok observations",
        "seed": seed,
        "train_trip_count": len(train_trips),
        "student_test_trip_ids": list(student_trips),
        "folds": splits,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_config": asdict(config),
        "label_source": LABEL_SOURCE,
        "label_config": asdict(LabelConfig()),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
        "search": {
            "method": "Gaussian process (Matern 2.5) with expected improvement",
            "log10_C_bounds": [MIN_LOG_C, MAX_LOG_C],
            "log2_positive_class_weight_bounds": [MIN_LOG_WEIGHT, MAX_LOG_WEIGHT],
            "grid_size_per_axis": GRID_SIZE,
            "initial_grid_indices": INITIAL_GRID_INDICES,
            "surrogate_length_scale": LENGTH_SCALE,
            "surrogate_alpha": SURROGATE_ALPHA,
            "tie_break": "earliest trial",
        },
        "train_preparation_seconds": train_preparation_seconds,
        "search_seconds": search_seconds,
        "student_test_preparation_seconds": student_preparation_seconds,
        "wall_seconds": perf_counter() - started,
        "best_trial": best["number"],
        "trials": trials,
        "comparison": comparison,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Best trial {best['number'] + 1}/{n_trials}. Results saved to {out}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Path to the Student folder.")
    parser.add_argument("--out", type=Path, default=Path("reports/model-tuning"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("Expected at least two folds.")
    if not len(INITIAL_GRID_INDICES) < args.trials <= GRID_SIZE ** 2:
        parser.error(f"Expected between {len(INITIAL_GRID_INDICES) + 1} and {GRID_SIZE ** 2} trials.")
    try:
        run(args.root, args.out, args.folds, args.seed, args.trials)
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
