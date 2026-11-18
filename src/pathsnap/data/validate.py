"""Data-quality checks for a single trip. Returns plain dicts so results can go to CSV."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .reader import Trip

LARGE_GAP_S = 5.0  # threshold for "large" gaps in the summary; nominal sampling is 1 s


def _coord_check(lon: np.ndarray, lat: np.ndarray) -> dict:
    finite = np.isfinite(lon) & np.isfinite(lat)
    in_range = finite & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    return {
        "n_nonfinite": int((~finite).sum()),
        "n_out_of_range": int((finite & ~in_range).sum()),
        "lon_min": float(np.nanmin(lon)) if len(lon) else np.nan,
        "lon_max": float(np.nanmax(lon)) if len(lon) else np.nan,
        "lat_min": float(np.nanmin(lat)) if len(lat) else np.nan,
        "lat_max": float(np.nanmax(lat)) if len(lat) else np.nan,
    }


def validate_trip(trip: Trip) -> dict:
    obs, nodes, arcs = trip.observations, trip.nodes, trip.arcs
    out: dict = {"trip_id": trip.trip_id, "partition": trip.partition}
    for ext, path in sorted(trip.source_files.items()):
        out[f"bytes_{ext}"] = path.stat().st_size

    # --- GPS track
    out["n_obs"] = len(obs)
    for k, v in _coord_check(obs["lon"].to_numpy(), obs["lat"].to_numpy()).items():
        out[f"track_{k}"] = v
    t = obs["t"].to_numpy()
    dt = np.diff(t)
    out["track_n_nan_time"] = int(np.isnan(t).sum())
    out["track_n_nonincreasing"] = int((dt <= 0).sum())
    out["track_n_duplicate_time"] = int((dt == 0).sum())
    out["track_duration_s"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out["track_dt_median_s"] = float(np.median(dt)) if len(dt) else np.nan
    out["track_dt_max_s"] = float(dt.max()) if len(dt) else np.nan
    out["track_frac_dt_eq_1s"] = float(np.mean(dt == 1.0)) if len(dt) else np.nan
    out["track_n_gaps_gt_1s"] = int((dt > 1.0).sum())
    out[f"track_n_gaps_gt_{int(LARGE_GAP_S)}s"] = int((dt > LARGE_GAP_S).sum())

    # --- Nodes / arcs
    n_nodes = len(nodes)
    out["n_nodes"] = n_nodes
    out["n_arcs"] = len(arcs)
    for k, v in _coord_check(nodes["lon"].to_numpy(), nodes["lat"].to_numpy()).items():
        out[f"nodes_{k}"] = v
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    bad_ref = (fr < 0) | (fr >= n_nodes) | (to < 0) | (to >= n_nodes)
    out["arcs_n_invalid_node_ref"] = int(bad_ref.sum())
    out["arcs_n_self_loops"] = int((fr == to).sum())
    pairs = np.stack([fr, to], axis=1)
    out["arcs_n_duplicate_pairs"] = int(len(pairs) - len(np.unique(pairs, axis=0))) if len(pairs) else 0
    # how many arcs have their reverse also present (two-way roads stored as two arcs?)
    pair_set = {(a, b) for a, b in pairs.tolist()}
    out["arcs_n_with_reverse"] = int(sum((b, a) in pair_set for a, b in pairs.tolist()))

    ok = ~bad_ref
    used = np.zeros(n_nodes, dtype=bool)
    used[fr[ok]] = True
    used[to[ok]] = True
    out["nodes_n_isolated"] = int((~used).sum())
    if n_nodes:
        g = coo_matrix((np.ones(ok.sum()), (fr[ok], to[ok])), shape=(n_nodes, n_nodes))
        n_weak, lab_w = connected_components(g, directed=True, connection="weak")
        n_strong, lab_s = connected_components(g, directed=True, connection="strong")
        out["graph_n_weak_components_incl_isolated"] = int(n_weak)
        out["graph_largest_weak_component_nodes"] = int(np.bincount(lab_w).max())
        out["graph_n_strong_components"] = int(n_strong)
        out["graph_largest_strong_component_nodes"] = int(np.bincount(lab_s).max())

    # --- Reference route (reference information only)
    if trip.route is not None:
        r = trip.route
        out["n_route"] = len(r)
        out["route_n_invalid_arc_ref"] = int(((r < 0) | (r >= len(arcs))).sum())
        valid = r[(r >= 0) & (r < len(arcs))]
        out["route_n_unique_arcs"] = int(len(np.unique(valid)))
        out["route_n_consecutive_repeats"] = int((np.diff(valid) == 0).sum()) if len(valid) > 1 else 0
        if len(valid) > 1:
            breaks = to[valid[:-1]] != fr[valid[1:]]
            out["route_n_discontinuities"] = int(breaks.sum())
        else:
            out["route_n_discontinuities"] = 0
    else:
        out["n_route"] = None
    return out
