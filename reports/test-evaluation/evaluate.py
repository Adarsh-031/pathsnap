"""Compare the saved tuned PathSnap model on both held-out partitions."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
from itertools import chain
from math import floor
from pathlib import Path
from time import perf_counter

import matplotlib
import pandas as pd

from pathsnap.data.labels import derive_observation_labels
from pathsnap.data.reader import discover_trips, load_trip_files, read_route
from pathsnap.features.candidates import build_candidates
from pathsnap.models.artifact import load_model_artifact
from pathsnap.reconstruction.hmm import HMMConfig, candidate_probabilities, decode_route
from pathsnap.reconstruction.road_graph import RoadGraph

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


PARTITIONS = ("student_test", "evaluator_test")
METHODS = {
    "nearest": ("Nearest candidate", "#778494"),
    "top1": ("Classifier top-1", "#a34a12"),
    "hmm": ("Classifier + HMM", "#006d77"),
}


def ordered_f1(predicted, reference):
    """Return twice the longest common subsequence length over both lengths."""
    previous = [0] * (len(reference) + 1)
    for arc_id in predicted:
        current = [0]
        for index, reference_id in enumerate(reference, start=1):
            if arc_id == reference_id:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return 2 * previous[-1] / (len(predicted) + len(reference))


def evaluate_trip(partition, trip_id, files, classifier, candidate_config, hmm_config):
    trip = load_trip_files(trip_id, partition, files, include_route=False)
    observations, candidates = build_candidates(trip, candidate_config)
    graph = RoadGraph(trip)
    started = perf_counter()
    probabilities = candidate_probabilities(classifier, candidates)
    decoded = decode_route(graph, observations, candidates, probabilities, config=hmm_config)
    inference_ms = (perf_counter() - started) * 1000

    # Reference files are opened after prediction.
    reference = read_route(files["route"])
    labels = derive_observation_labels(replace(trip, route=reference))
    scorable = labels.loc[labels["label_status"].eq("ok")].set_index("observation_index")
    if scorable.empty:
        raise ValueError(f"{partition}/{trip_id} has no scorable observations")

    ranked = candidates.assign(probability=probabilities).sort_values("cand_rank", kind="stable")
    winners = ranked.loc[ranked.groupby("observation_index")["probability"].idxmax()]
    top1 = winners.set_index("observation_index")["arc_id"].astype("Int64")
    nearest = ranked.loc[ranked["cand_rank"].eq(0)].set_index("observation_index")["arc_id"]
    hmm = pd.Series({match.observation_index: match.arc_id for match in decoded.matches},
                    dtype="Int64")

    steps = observations.set_index("observation_index").join(top1)
    new_arc = steps["arc_id"].ne(steps["arc_id"].shift()).fillna(True)
    top1_sequence = steps.loc[steps["arc_id"].notna() & (new_arc | steps["gap_flag"].eq(1)), "arc_id"]
    hmm_sequence = list(chain.from_iterable(decoded.sections))

    row = {
        "partition": partition,
        "trip_id": trip_id,
        "observations": len(labels),
        "scorable": len(scorable),
        "ambiguous": int(labels["label_status"].eq("ambiguous").sum()),
        "far": int(labels["label_status"].eq("far").sum()),
    }
    for method, predicted in (("nearest", nearest), ("top1", top1), ("hmm", hmm)):
        matches = predicted.reindex(scorable.index).eq(scorable["label_arc_id"])
        row[f"{method}_correct"] = int(matches.sum())
        row[f"{method}_accuracy"] = row[f"{method}_correct"] / len(scorable)
    row.update(
        top1_ordered_route_f1=ordered_f1(top1_sequence.tolist(), reference.tolist()),
        hmm_ordered_route_f1=ordered_f1(hmm_sequence, reference.tolist()),
        hmm_sections=len(decoded.sections),
        inference_ms=inference_ms,
    )
    return row


def summarize(rows):
    scorable = int(rows["scorable"].sum())
    return {
        "trips": len(rows),
        "observations": int(rows["observations"].sum()),
        "scorable": scorable,
        "ambiguous": int(rows["ambiguous"].sum()),
        "far": int(rows["far"].sum()),
        "methods": {
            method: {
                "correct": int(rows[f"{method}_correct"].sum()),
                "point_weighted_accuracy": float(rows[f"{method}_correct"].sum() / scorable),
                "mean_trip_accuracy": float(rows[f"{method}_accuracy"].mean()),
            }
            for method in METHODS
        },
        "mean_ordered_route_f1": {
            method: float(rows[f"{method}_ordered_route_f1"].mean())
            for method in ("top1", "hmm")
        },
        "median_inference_ms": float(rows["inference_ms"].median()),
    }


def plot_overview(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, title, ylabel, methods in (
        (axes[0], "Exact directed-arc accuracy", "Scorable GPS points matched",
         tuple(METHODS)),
        (axes[1], "Ordered route F1", "Mean F1 across trips", ("top1", "hmm")),
    ):
        for index, method in enumerate(methods):
            label, color = METHODS[method]
            if ax is axes[0]:
                scores = [summary[p]["methods"][method]["point_weighted_accuracy"]
                          for p in PARTITIONS]
            else:
                scores = [summary[p]["mean_ordered_route_f1"][method] for p in PARTITIONS]
            offsets = [i + (index - (len(methods) - 1) / 2) * 0.25 for i in range(len(PARTITIONS))]
            bars = ax.bar(offsets, scores, width=0.22, label=label, color=color)
            ax.bar_label(bars, labels=[f"{score:.1%}" for score in scores], padding=3, fontsize=9)
        ax.set(title=title, ylabel=ylabel, ylim=(0, 1.11))
        ax.set_xticks(range(len(PARTITIONS)), PARTITIONS)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(out / "partition-comparison.png", dpi=160)
    plt.close(fig)


def plot_trips(rows, out):
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5.2), layout="constrained")
    minimum = rows[["top1_accuracy", "hmm_accuracy"]].min().min()
    for ax, partition in zip(axes, PARTITIONS, strict=True):
        trips = rows.loc[rows["partition"].eq(partition)].sort_values("hmm_accuracy")
        y = range(len(trips))
        ax.hlines(y, trips["top1_accuracy"], trips["hmm_accuracy"], color="#c5ccd4", linewidth=1.5)
        for method, marker in (("top1", "o"), ("hmm", "D")):
            label, color = METHODS[method]
            face = "none" if method == "top1" else color
            size = 64 if method == "top1" else 24
            ax.scatter(trips[f"{method}_accuracy"], y, marker=marker, facecolors=face,
                       edgecolors=color, label=label, s=size, zorder=3)
        ax.set_yticks(y, trips["trip_id"])
        ax.invert_yaxis()
        ax.set(title=partition, xlabel="Scorable GPS points matched",
               xlim=(floor(10 * minimum) / 10, 1.025))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x", color="#e3e7eb")
    axes[0].set_ylabel("Trip ID")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Exact directed-arc accuracy by trip")
    fig.savefig(out / "trip-accuracy.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-root", type=Path, required=True, help="Root containing student_test")
    parser.add_argument("--evaluator-root", type=Path, required=True, help="Root containing evaluator_test")
    parser.add_argument("--model", type=Path, required=True, help="Trusted .joblib beside tuning metrics.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    manifests = {
        partition: discover_trips(root, partition=partition)
        for partition, root in zip(PARTITIONS, (args.student_root, args.evaluator_root), strict=True)
    }
    training = json.loads(args.model.with_name("metrics.json").read_text())
    train_ids = {trip_id for fold in training["folds"] for trip_id in fold["fit_trip_ids"]}
    student_ids, evaluator_ids = (set(manifests[partition]) for partition in PARTITIONS)
    if student_ids != set(training["student_test_trip_ids"]) or not evaluator_ids:
        parser.error("Expected the recorded student-test trips and a nonempty evaluator partition")
    if (student_ids & evaluator_ids) or ((student_ids | evaluator_ids) & train_ids):
        parser.error("Train, student-test and evaluator-test trip IDs must be disjoint")

    classifier, candidate_config = load_model_artifact(args.model)
    hmm_config = HMMConfig()
    rows = []
    for partition, manifest in manifests.items():
        for trip_id, info in manifest.items():
            print(f"Evaluating {partition}/{trip_id}", flush=True)
            rows.append(evaluate_trip(partition, trip_id, info["files"], classifier,
                                      candidate_config, hmm_config))
    results = pd.DataFrame(rows)
    summary = {partition: summarize(group)
               for partition, group in results.groupby("partition", sort=False)}
    report = {
        "artifact": args.model.name,
        "artifact_sha256": sha256(args.model.read_bytes()).hexdigest(),
        "training_trips": len(train_ids),
        "versions": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "pyproj", "scikit-learn", "joblib", "matplotlib",
        )},
        "candidate_config": asdict(candidate_config),
        "hmm_config": asdict(hmm_config),
        "partitions": summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "per-trip.csv", index=False, float_format="%.8f")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_overview(summary, args.out)
    plot_trips(results, args.out)


if __name__ == "__main__":
    main()
