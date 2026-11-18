"""DERIVED observation labels from the reference route (training / student-test only).

The supplied ground truth is ``.route``: an ordered list of arc IDs. It is NOT one
arc per GPS observation (e.g. 1,315 observations vs 525 route entries), so rows must
never be paired by position. Observation-level labels are *derived* here and are our
team's assumption, not organiser-supplied truth.

Method (``derive_observation_labels``), "monotone route alignment":
    Assign each observation i a route position j(i) such that j is non-decreasing in i
    (the vehicle never goes backwards along its own route) and the total distance
    sum_i dist(obs_i, arc route[j(i)]) is minimal. Solved exactly by dynamic programming,
    O(N * M). Route positions (not arc IDs) are aligned, so loops / revisited arcs are fine.
    Route arcs may receive zero observations (short arcs passed between samples).

Every label carries quality flags so evaluation can exclude or report uncertain points.
Never import this module from inference / API prediction code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathsnap.features.candidates import TripGeometry
from .reader import Trip

LABEL_SOURCE = "derived_monotone_route_alignment_v1"


@dataclass(frozen=True)
class LabelConfig:
    far_threshold_m: float = 30.0      # aligned arc farther than this -> status "far"
    ambiguity_margin_m: float = 3.0    # alternatives within this margin -> flagged
    alt_search_radius_m: float = 60.0  # search radius for competing non-route arcs


def _monotone_alignment(cost: np.ndarray) -> np.ndarray:
    """argmin over non-decreasing paths j(0) <= j(1) <= ... of sum cost[i, j(i)]."""
    n, m = cost.shape
    back = np.empty((n, m), dtype=np.int32)
    acc = cost[0].copy()
    back[0] = np.arange(m)
    cols = np.arange(m)
    for i in range(1, n):
        # vectorised prefix minimum of acc and the index where it occurs
        pm = np.minimum.accumulate(acc)
        is_new = np.r_[True, acc[1:] < pm[:-1]]
        back[i] = np.maximum.accumulate(np.where(is_new, cols, 0))
        acc = cost[i] + pm
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmin(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def derive_observation_labels(trip: Trip, cfg: LabelConfig = LabelConfig()) -> pd.DataFrame:
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no reference route; cannot derive labels")
    route = np.asarray(trip.route, dtype=np.int64)
    n_arcs = len(trip.arcs)
    if ((route < 0) | (route >= n_arcs)).any():
        raise ValueError(f"Trip {trip.trip_id}: route references unknown arc IDs")

    geom = TripGeometry.from_trip(trip.without_route())
    n = len(trip.observations)
    m = len(route)
    obs_rep = np.repeat(np.arange(n), m)
    d_route, _ = geom.distances(obs_rep, np.tile(route, n))
    d_route = d_route.reshape(n, m)                      # (N, M) over route positions
    pos = _monotone_alignment(d_route)
    lab_arc = route[pos]
    rows = np.arange(n)
    lab_dist = d_route[rows, pos]

    # Repeated positions for the same arc ID cannot change the observation's label.
    prev_pos = np.maximum(pos - 1, 0)
    next_pos = np.minimum(pos + 1, m - 1)
    prev_d = np.where(route[prev_pos] != lab_arc, d_route[rows, prev_pos], np.inf)
    next_d = np.where(route[next_pos] != lab_arc, d_route[rows, next_pos], np.inf)
    near_boundary = (np.minimum(prev_d, next_d) - lab_dist) <= cfg.ambiguity_margin_m

    # Non-adjacent arcs can also pass near the observation, e.g. on a loop.
    route_alt_min = np.min(d_route, axis=1, where=route != lab_arc[:, None], initial=np.inf)
    route_alt_close = (route_alt_min - lab_dist) <= cfg.ambiguity_margin_m

    # nearest arc that is NOT on the route and NOT the reverse twin of the label arc
    fr = trip.arcs["from_node"].to_numpy()
    to = trip.arcs["to_node"].to_numpy()
    on_route = np.zeros(len(fr), dtype=bool)
    on_route[route] = True
    alt_min = np.full(n, np.inf)
    o, a, d, _ = geom.build_index().query(rows, cfg.alt_search_radius_m)
    twin = (fr[a] == to[lab_arc[o]]) & (to[a] == fr[lab_arc[o]])
    ok = ~on_route[a] & ~twin
    np.minimum.at(alt_min, o[ok], d[ok])
    alt_close = (alt_min - lab_dist) <= cfg.ambiguity_margin_m

    # The all-route comparison includes the adjacent arcs flagged by near_boundary.
    status = np.where(lab_dist > cfg.far_threshold_m, "far",
                      np.where(route_alt_close | alt_close, "ambiguous", "ok"))
    return pd.DataFrame({
        "trip_id": trip.trip_id,
        "observation_index": trip.observations["observation_index"].to_numpy(),
        "label_arc_id": lab_arc,
        "route_position": pos,
        "label_dist_m": lab_dist,
        "nearest_nonroute_dist_m": alt_min,
        "near_route_boundary": near_boundary.astype(int),
        "label_status": status,
        "label_source": LABEL_SOURCE,
    })


def label_summary(trip: Trip, labels: pd.DataFrame) -> dict:
    m = len(trip.route)
    covered = np.zeros(m, dtype=bool)
    covered[labels["route_position"].to_numpy()] = True
    st = labels["label_status"].value_counts()
    return {
        "trip_id": trip.trip_id,
        "n_obs": len(labels),
        "n_route": m,
        "label_ok": int(st.get("ok", 0)),
        "label_ambiguous": int(st.get("ambiguous", 0)),
        "label_far": int(st.get("far", 0)),
        "near_route_boundary": int(labels["near_route_boundary"].sum()),
        "route_positions_with_no_obs": int((~covered).sum()),
        "label_dist_median_m": float(labels["label_dist_m"].median()),
        "label_dist_p95_m": float(labels["label_dist_m"].quantile(0.95)),
        "label_dist_max_m": float(labels["label_dist_m"].max()),
    }


def candidate_coverage(candidates: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Candidate miss rate: labelled observations whose label arc is not among candidates."""
    key = ["trip_id", "observation_index"]
    hit = candidates.merge(labels[key + ["label_arc_id"]], on=key)
    hit = hit[hit["arc_id"] == hit["label_arc_id"]][key + ["cand_rank"]]
    j = labels.merge(hit, on=key, how="left")
    out = {"n_labelled": len(j)}
    for name, mask in [("all", np.ones(len(j), bool)), ("ok_only", (j["label_status"] == "ok").to_numpy())]:
        sub = j[mask]
        miss = sub["cand_rank"].isna()
        out[f"miss_count_{name}"] = int(miss.sum())
        out[f"miss_rate_{name}"] = float(miss.mean()) if len(sub) else np.nan
        out[f"label_is_rank0_{name}"] = float((sub["cand_rank"] == 0).mean()) if len(sub) else np.nan
    return out


def attach_training_target(candidates: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Binary target y for candidate rows (1 = candidate arc equals derived label arc).

    Returned as a separate Series aligned to ``candidates`` so it never sits in the
    feature matrix. Training / student-test evaluation only.
    """
    key = ["trip_id", "observation_index"]
    lab = candidates[key].merge(labels[key + ["label_arc_id"]], on=key, how="left")
    return pd.Series((candidates["arc_id"].to_numpy() == lab["label_arc_id"].to_numpy()).astype(int),
                     index=candidates.index, name="y")
