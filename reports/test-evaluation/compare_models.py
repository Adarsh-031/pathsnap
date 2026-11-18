"""Compare all five fitted classifiers with and without HMM on student_test."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

import pandas as pd

from evaluate import score_trip
from pathsnap.data.labels import derive_observation_labels
from pathsnap.data.reader import discover_trips, load_trip_files, read_route
from pathsnap.features.candidates import CandidateConfig, build_candidates
from pathsnap.models.artifact import load_model_artifact
from pathsnap.reconstruction.hmm import HMMConfig, candidate_probabilities, decode_route
from pathsnap.reconstruction.road_graph import RoadGraph


PARTITION = "student_test"
MODEL_NAMES = {
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}


def evaluate_models(manifest, classifiers, candidate_config, hmm_config):
    rows = []
    for trip_id, info in sorted(manifest.items()):
        print(f"Evaluating {PARTITION}/{trip_id} with all five models", flush=True)
        files = info["files"]
        trip = load_trip_files(trip_id, PARTITION, files, include_route=False)
        observations, candidates = build_candidates(trip, candidate_config)
        graph = RoadGraph(trip)

        predictions = []
        for name, classifier in classifiers.items():
            started = perf_counter()
            probabilities = candidate_probabilities(classifier, candidates)
            scored = perf_counter()
            decoded = decode_route(graph, observations, candidates, probabilities, config=hmm_config)
            finished = perf_counter()
            predictions.append((name, probabilities, decoded, scored - started, finished - scored))

        reference = read_route(files["route"])
        labels = derive_observation_labels(replace(trip, route=reference))
        for name, probabilities, decoded, scoring_seconds, decoding_seconds in predictions:
            row = score_trip(PARTITION, trip_id, observations, candidates, probabilities,
                             decoded, reference, labels, graph=graph)
            rows.append({"model": name, **row, "classifier_scoring_ms": scoring_seconds * 1000,
                         "hmm_decode_ms": decoding_seconds * 1000})
    return pd.DataFrame(rows)


def summarize_model(name, rows, training):
    summary = {
        "model": name,
        "trips": len(rows),
        "scorable_observations": int(rows["scorable"].sum()),
        "training_cv_mean_top1_accuracy": fmean(
            fold["ok"]["top1_accuracy"] for fold in training["cross_validation"]
        ),
        "training_cv_mean_candidate_macro_f1": fmean(
            fold["ok"]["classification"]["macro_f1"] for fold in training["cross_validation"]
        ),
        "student_candidate_macro_f1": training["student_test"]["ok"]["classification"]["macro_f1"],
        "student_candidate_log_loss": training["student_test"]["ok"]["classification"]["log_loss"],
    }
    for method in ("top1", "hmm"):
        paired_steps = int(rows[f"{method}_direct_pairs"].sum())
        summary.update({
            f"{method}_exact_arc_accuracy": rows[f"{method}_correct"].sum() / summary["scorable_observations"],
            f"{method}_mean_ordered_route_f1": rows[f"{method}_ordered_route_f1"].mean(),
            f"{method}_direct_continuity": (
                rows[f"{method}_direct_connected"].sum() / paired_steps if paired_steps else None
            ),
        })
    summary["median_classifier_scoring_ms"] = rows["classifier_scoring_ms"].median()
    summary["median_classifier_and_hmm_ms"] = (rows["classifier_scoring_ms"] + rows["hmm_decode_ms"]).median()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-root", type=Path, required=True, help="Directory containing student_test")
    parser.add_argument("--model-dir", type=Path, required=True, help="Trusted training outputs with metrics.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "five-models")
    args = parser.parse_args()

    training = json.loads((args.model_dir / "metrics.json").read_text())
    manifest = discover_trips(args.student_root, partition=PARTITION)
    train_ids = set(training["train_trip_ids"])
    if not manifest or set(manifest) != set(training["student_test_trip_ids"]) or train_ids & manifest.keys():
        parser.error("Expected the recorded student-test trip IDs, disjoint from training")

    candidate_config = CandidateConfig(**training["candidate_config"])
    hmm_config = HMMConfig()
    classifiers = {}
    artifacts = {}
    for key, name in MODEL_NAMES.items():
        artifact = args.model_dir / training["models"][key]["artifact"]
        classifier, config = load_model_artifact(artifact)
        if config != candidate_config:
            parser.error(f"{artifact.name} disagrees with the training run's candidate settings")
        classifiers[name] = classifier
        artifacts[name] = {"file": artifact.name, "sha256": sha256(artifact.read_bytes()).hexdigest()}

    rows = evaluate_models(manifest, classifiers, candidate_config, hmm_config)
    summary = pd.DataFrame([
        summarize_model(name, rows.loc[rows["model"].eq(name)], training["models"][key])
        for key, name in MODEL_NAMES.items()
    ])
    metadata = {
        "partition": PARTITION,
        "trip_ids": sorted(manifest),
        "training_trips": len(train_ids),
        "training_seed": training["seed"],
        "training_folds": len(training["folds"]),
        "training_versions": training["versions"],
        "evaluation_versions": {name: version(name) for name in (*training["versions"], "scipy", "pyproj")},
        "candidate_config": asdict(candidate_config),
        "hmm_config": asdict(hmm_config),
        "artifacts": artifacts,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "per-trip.csv", index=False, float_format="%.8f")
    summary.to_csv(args.out / "by-model.csv", index=False, float_format="%.8f")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
