"""Candidate classification and observation ranking metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

OBS_KEY = ["trip_id", "observation_index"]


def _rate(flags: pd.Series) -> float | None:
    return float(flags.mean()) if len(flags) else None


def evaluate(
    rows: pd.DataFrame, observations: pd.DataFrame, probabilities: np.ndarray,
) -> dict:
    scored = rows.assign(probability=probabilities).sort_values("cand_rank", kind="stable")
    # idxmax keeps the nearest candidate when probability scores tie.
    winners = scored.loc[scored.groupby(OBS_KEY, sort=False)["probability"].idxmax()]
    nearest = scored.loc[scored["cand_rank"].eq(0), OBS_KEY + ["arc_id"]]
    hits = scored.loc[scored["y"].eq(1), OBS_KEY].assign(candidate_hit=True)
    obs = observations.merge(
        winners[OBS_KEY + ["arc_id", "from_node", "to_node"]], on=OBS_KEY, how="left",
    ).merge(
        nearest.rename(columns={"arc_id": "nearest_arc_id"}), on=OBS_KEY, how="left",
    ).merge(hits, on=OBS_KEY, how="left")

    results = {}
    for status in ("all", "ok", "ambiguous", "far"):
        subset = obs if status == "all" else obs[obs["label_status"].eq(status)]
        results[status] = {
            "observations": len(subset),
            "top1_accuracy": _rate(subset["arc_id"].eq(subset["label_arc_id"])),
            "nearest_top1_accuracy": _rate(subset["nearest_arc_id"].eq(subset["label_arc_id"])),
            "candidate_miss_rate": _rate(subset["candidate_hit"].isna()),
            "no_candidate_count": int(subset["arc_id"].isna().sum()),
        }

    for status, subset in (("all", scored), ("ok", scored[scored["label_status"].eq("ok")])):
        if subset.empty:
            results[status]["classification"] = None
            continue
        y = subset["y"]
        p = subset["probability"].to_numpy()
        predictions = p >= 0.5
        results[status]["classification"] = {
            "candidate_rows": len(subset),
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
            "f1": f1_score(y, predictions, zero_division=0),
            "macro_f1": f1_score(y, predictions, labels=[0, 1], average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
            "log_loss": log_loss(y, p, labels=[0, 1]),
        }

    obs = obs.sort_values(OBS_KEY)
    previous = obs.groupby("trip_id", sort=False)[["arc_id", "to_node"]].shift()
    eligible = previous.index[previous["arc_id"].notna() & obs["arc_id"].notna()
                              & obs["gap_flag"].eq(0)]
    connected = obs.loc[eligible, "arc_id"].eq(previous.loc[eligible, "arc_id"]) | (
        obs.loc[eligible, "from_node"].eq(previous.loc[eligible, "to_node"])
    )
    results["direct_continuity"] = {"scored_pairs": len(connected), "rate": _rate(connected)}
    return results
