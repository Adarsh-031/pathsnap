"""Fit and compare the five PathSnap candidate classifiers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from pathsnap.data.labels import (
    LABEL_SOURCE,
    LabelConfig,
    attach_training_target,
    derive_observation_labels,
)
from pathsnap.data.reader import discover_trips, load_trip_files
from pathsnap.data.splits import DEV_PARTITIONS_FOR_FITTING, trip_group_folds
from pathsnap.features.candidates import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    CandidateConfig,
    build_candidates,
)

from .artifact import save_model_artifact
from .estimators import candidate_models
from .evaluation import OBS_KEY, evaluate


def prepare_partition(trips: dict[str, dict], trip_ids: list[str], config: CandidateConfig):
    observation_frames, candidate_frames = [], []
    for trip_id in trip_ids:
        print(f"Preparing trip {trip_id}", flush=True)
        info = trips[trip_id]
        trip = load_trip_files(trip_id, info["partition"], info["files"], include_route=True)
        observations, candidates = build_candidates(trip.without_route(), config)
        labels = derive_observation_labels(trip)
        reference = labels[OBS_KEY + ["label_arc_id", "label_status"]]
        observation_frames.append(reference.merge(observations[OBS_KEY + ["gap_flag"]], on=OBS_KEY))

        candidates["y"] = attach_training_target(candidates, labels)
        candidates = candidates.merge(reference[OBS_KEY + ["label_status"]], on=OBS_KEY)
        endpoints = trip.arcs[["arc_id", "from_node", "to_node"]]
        candidate_frames.append(candidates.merge(endpoints, on="arc_id"))
    return pd.concat(observation_frames, ignore_index=True), pd.concat(candidate_frames, ignore_index=True)


def fit_model(estimator, rows: pd.DataFrame) -> float:
    rows = rows[rows["label_status"].eq("ok")]
    y = rows["y"]
    if y.nunique() != 2:
        raise ValueError("Fitting requires positive and negative candidates from ok labels.")
    features = rows[FEATURE_COLUMNS]
    start = perf_counter()
    estimator.fit(features, y)
    return perf_counter() - start


def score_model(estimator, rows: pd.DataFrame, observations: pd.DataFrame) -> dict:
    features = rows[FEATURE_COLUMNS]
    start = perf_counter()
    probabilities = estimator.predict_proba(features)[:, 1] if len(features) else np.empty(0)
    elapsed = perf_counter() - start
    return {"inference_seconds": elapsed, **evaluate(rows, observations, probabilities)}


def model_parameters(estimator) -> dict:
    if isinstance(estimator, Pipeline):
        return {name: step.get_params(deep=False) for name, step in estimator.steps}
    return {
        key: "NaN" if isinstance(value, float) and np.isnan(value) else value
        for key, value in estimator.get_params(deep=False).items()
    }


def run(root: Path, out: Path, folds: int, seed: int, jobs: int, limit: int | None) -> None:
    if out.exists():
        raise ValueError(f"Output path {out} already exists. Choose a new --out path.")

    trips = discover_trips(root)
    train_ids = [trip_id for trip_id, info in trips.items()
                 if info["partition"] in DEV_PARTITIONS_FOR_FITTING][:limit]
    student_ids = [trip_id for trip_id, info in trips.items()
                   if info["partition"] == "student_test"][:limit]
    if len(train_ids) < folds or not student_ids:
        raise ValueError(f"Expected at least {folds} train trips and one student_test trip under {root}.")
    for trip_id in train_ids + student_ids:
        if "route" not in trips[trip_id]["files"]:
            raise ValueError(f"Trip {trip_id} is missing its reference route.")

    config = CandidateConfig()
    obs_train, rows_train = prepare_partition(trips, train_ids, config)
    obs_student, rows_student = prepare_partition(trips, student_ids, config)
    splits = []
    for fit_idx, val_idx in trip_group_folds(obs_train, n_splits=folds, seed=seed):
        fit_trips = sorted(obs_train.iloc[fit_idx]["trip_id"].unique())
        val_trips = sorted(obs_train.iloc[val_idx]["trip_id"].unique())
        splits.append((fit_trips, val_trips))

    run_info = {
        "seed": seed,
        "jobs": jobs,
        "trip_limit_per_partition": limit,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_config": asdict(config),
        "label_source": LABEL_SOURCE,
        "label_config": asdict(LabelConfig()),
        "train_trip_ids": train_ids,
        "student_test_trip_ids": student_ids,
        "folds": [{"fit_trip_ids": fit_trips, "validation_trip_ids": val_trips}
                  for fit_trips, val_trips in splits],
        "versions": {name: version(name) for name in (
            "numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "xgboost", "joblib",
        )},
    }
    out.mkdir(parents=True)
    results = {}
    for name, template in candidate_models(seed, jobs).items():
        print(f"Training {name}", flush=True)
        fold_results = []
        for fit_trips, val_trips in splits:
            estimator = clone(template)
            fit_time = fit_model(estimator, rows_train[rows_train["trip_id"].isin(fit_trips)])
            metrics = score_model(
                estimator, rows_train[rows_train["trip_id"].isin(val_trips)],
                obs_train[obs_train["trip_id"].isin(val_trips)],
            )
            fold_results.append({"fit_seconds": fit_time, **metrics})

        estimator = clone(template)
        fit_time = fit_model(estimator, rows_train)
        model_file = f"{name}.joblib"
        results[name] = {
            "parameters": model_parameters(template),
            "cross_validation": fold_results,
            "refit_seconds": fit_time,
            "student_test": score_model(estimator, rows_student, obs_student),
            "artifact": model_file,
        }
        save_model_artifact(out / model_file, estimator, config)

    with (out / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({**run_info, "models": results}, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Path to the Student folder.")
    parser.add_argument("--out", type=Path, default=Path("reports/model-training"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4, help="Worker count for each tree model.")
    parser.add_argument("--trip-limit-per-partition", type=int, help="Limit trips for a smoke run.")
    args = parser.parse_args()
    if args.folds < 2 or args.jobs < 1:
        parser.error("Expected at least two folds and a positive worker count.")
    if args.trip_limit_per_partition is not None and args.trip_limit_per_partition < args.folds:
        parser.error("--trip-limit-per-partition must be at least --folds.")
    try:
        run(args.root, args.out, args.folds, args.seed, args.jobs, args.trip_limit_per_partition)
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
