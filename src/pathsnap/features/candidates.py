"""Inference-safe candidate generation and feature extraction.

Nothing in this module reads the reference route. ``build_candidates`` asserts that.
Outputs:
    observations table  (one row per GPS observation, including ones with no candidates)
    candidates table    (one row per (observation, candidate arc)), columns = KEY_COLUMNS + FEATURE_COLUMNS
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pathsnap.data.reader import Trip
from .geometry import Projector, angle_diff_deg, bearing_deg, point_segment

FEATURE_SCHEMA_VERSION = "1.0"
KEY_COLUMNS = ["trip_id", "observation_index", "arc_id"]
# Stable, ordered. Append new features at the end and bump FEATURE_SCHEMA_VERSION.
FEATURE_COLUMNS = [
    "dist_m",              # perpendicular distance GPS point -> arc segment (metres)
    "dist_minus_min_m",    # dist_m minus the nearest candidate's dist_m for this observation
    "cand_rank",           # 0 = nearest candidate
    "n_candidates",        # candidates kept for this observation
    "proj_frac",           # 0..1 position of the projected point along the arc (from -> to)
    "arc_length_m",
    "heading_diff_deg",    # |GPS heading - arc direction| in [0,180]; HEADING_FILL if heading_valid == 0
    "heading_valid",       # 1 if the GPS heading is reliable (vehicle moved enough), else 0
    "speed_mps",           # speed from previous observation (0 for the first)
    "dt_prev_s",           # time since previous observation (0 for the first)
    "gap_flag",            # 1 if dt_prev_s > gap_threshold_s
    "from_node_in_degree",
    "to_node_out_degree",
]
HEADING_FILL = 90.0  # constant (not learned) fill value: "no direction information"


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 50.0
    max_candidates: int = 8
    heading_min_move_m: float = 2.0
    gap_threshold_s: float = 5.0


@dataclass
class TripGeometry:
    """Projected geometry for one trip (no reference data)."""

    projector: Projector
    obs_xy: np.ndarray       # (N, 2)
    node_xy: np.ndarray      # (n_nodes, 2)
    arc_a: np.ndarray        # (n_arcs, 2) start points
    arc_b: np.ndarray        # (n_arcs, 2) end points
    arc_length: np.ndarray
    arc_bearing: np.ndarray

    @classmethod
    def from_trip(cls, trip: Trip) -> "TripGeometry":
        obs, nodes, arcs = trip.observations, trip.nodes, trip.arcs
        proj = Projector.for_lonlat(obs["lon"].to_numpy(), obs["lat"].to_numpy())
        ox, oy = proj.to_xy(obs["lon"], obs["lat"])
        nx, ny = proj.to_xy(nodes["lon"], nodes["lat"])
        node_xy = np.column_stack([nx, ny])
        a = node_xy[arcs["from_node"].to_numpy()]
        b = node_xy[arcs["to_node"].to_numpy()]
        d = b - a
        return cls(proj, np.column_stack([ox, oy]), node_xy, a, b,
                   np.hypot(d[:, 0], d[:, 1]), bearing_deg(d[:, 0], d[:, 1]))

    def distances(self, obs_idx: np.ndarray, arc_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Elementwise distance/projection fraction for paired (observation, arc) index arrays."""
        p = self.obs_xy[obs_idx]
        a, b = self.arc_a[arc_idx], self.arc_b[arc_idx]
        return point_segment(p[:, 0], p[:, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])

    def build_index(self, sample_step_m: float = 10.0) -> "ArcIndex":
        return ArcIndex(self, sample_step_m)


class ArcIndex:
    """KD-tree over points sampled every <= step metres along every arc.

    Any point on an arc lies within step/2 of a sample, so querying samples with
    radius R + step/2 returns a superset of the arcs within R; exact
    point-to-segment distances are then computed for that superset only.
    """

    def __init__(self, geom: TripGeometry, step: float):
        self.geom, self.step = geom, step
        n_seg = np.maximum(1, np.ceil(geom.arc_length / step).astype(int))
        owner = np.repeat(np.arange(len(n_seg)), n_seg + 1)
        start = np.repeat(np.cumsum(n_seg + 1) - (n_seg + 1), n_seg + 1)
        frac = (np.arange(len(owner)) - start) / np.repeat(n_seg, n_seg + 1)
        pts = geom.arc_a[owner] + frac[:, None] * (geom.arc_b[owner] - geom.arc_a[owner])
        self.owner = owner
        self.tree = cKDTree(pts)

    def query(self, obs_idx: np.ndarray, radius_m: float):
        """Return (obs, arc, dist, frac) arrays for every arc within radius_m of each observation."""
        hits = self.tree.query_ball_point(self.geom.obs_xy[obs_idx], r=radius_m + self.step / 2)
        o_list, a_list = [], []
        for i, h in zip(obs_idx, hits):
            if h:
                arcs = np.unique(self.owner[h])
                o_list.append(np.full(arcs.size, i))
                a_list.append(arcs)
        if not o_list:
            e = np.empty(0)
            return e.astype(int), e.astype(int), e, e
        o, a = np.concatenate(o_list), np.concatenate(a_list)
        d, f = self.geom.distances(o, a)
        keep = d <= radius_m
        return o[keep], a[keep], d[keep], f[keep]


def observation_table(trip: Trip, geom: TripGeometry, cfg: CandidateConfig) -> pd.DataFrame:
    obs = trip.observations
    xy, t = geom.obs_xy, obs["t"].to_numpy()
    n = len(obs)
    dt_prev = np.zeros(n)
    step = np.zeros(n)
    if n > 1:
        dt_prev[1:] = np.diff(t)
        step[1:] = np.hypot(*np.diff(xy, axis=0).T)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt_prev > 0, step / dt_prev, 0.0)
    # heading: central difference over neighbours (uses i-1 and i+1; full trip is known offline)
    prev_i = np.clip(np.arange(n) - 1, 0, n - 1)
    next_i = np.clip(np.arange(n) + 1, 0, n - 1)
    d = xy[next_i] - xy[prev_i]
    move = np.hypot(d[:, 0], d[:, 1])
    heading = bearing_deg(d[:, 0], d[:, 1])
    valid = move >= cfg.heading_min_move_m
    return pd.DataFrame({
        "trip_id": trip.trip_id,
        "observation_index": obs["observation_index"].to_numpy(),
        "lon": obs["lon"].to_numpy(), "lat": obs["lat"].to_numpy(), "t": t,
        "x_m": xy[:, 0], "y_m": xy[:, 1],
        "dt_prev_s": dt_prev, "speed_mps": speed,
        "heading_deg": np.where(valid, heading, np.nan), "heading_valid": valid.astype(int),
        "gap_flag": (dt_prev > cfg.gap_threshold_s).astype(int),
    })


def build_candidates(trip: Trip, cfg: CandidateConfig = CandidateConfig()
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (observations, candidates) for one trip without using its route."""
    trip = trip.without_route()  # guarantees no reference information is reachable here
    geom = TripGeometry.from_trip(trip)
    obs_tab = observation_table(trip, geom, cfg)
    arcs = trip.arcs
    n_nodes = len(trip.nodes)
    in_deg = np.bincount(arcs["to_node"], minlength=n_nodes)
    out_deg = np.bincount(arcs["from_node"], minlength=n_nodes)
    arc_ids = arcs["arc_id"].to_numpy()

    n = len(obs_tab)
    index = geom.build_index()
    finite = np.isfinite(geom.obs_xy).all(axis=1)
    o, a_idx, d, f = index.query(np.flatnonzero(finite), cfg.radius_m)
    c = pd.DataFrame({"observation_index": o, "arc_id": arc_ids[a_idx], "dist_m": d, "proj_frac": f})
    c = c.sort_values(["observation_index", "dist_m", "arc_id"], kind="mergesort")
    c["cand_rank"] = c.groupby("observation_index").cumcount()
    c = c[c["cand_rank"] < cfg.max_candidates].reset_index(drop=True)
    n_cand = np.bincount(c["observation_index"].to_numpy(int), minlength=n)
    obs_tab["n_candidates"] = n_cand
    obs_tab["has_candidates"] = (n_cand > 0).astype(int)

    c["trip_id"] = trip.trip_id
    c["dist_minus_min_m"] = c["dist_m"] - c.groupby("observation_index")["dist_m"].transform("min")
    c["n_candidates"] = n_cand[c["observation_index"].to_numpy(int)] if len(c) else 0
    a = c["arc_id"].to_numpy(int)
    c["arc_length_m"] = geom.arc_length[a]
    o = obs_tab.set_index("observation_index").loc[c["observation_index"]]
    hv = o["heading_valid"].to_numpy()
    hd = angle_diff_deg(o["heading_deg"].fillna(0).to_numpy(), geom.arc_bearing[a])
    c["heading_diff_deg"] = np.where(hv == 1, hd, HEADING_FILL)
    c["heading_valid"] = hv
    c["speed_mps"] = o["speed_mps"].to_numpy()
    c["dt_prev_s"] = o["dt_prev_s"].to_numpy()
    c["gap_flag"] = o["gap_flag"].to_numpy()
    c["from_node_in_degree"] = in_deg[arcs["from_node"].to_numpy()[a]]
    c["to_node_out_degree"] = out_deg[arcs["to_node"].to_numpy()[a]]
    c = c[KEY_COLUMNS + FEATURE_COLUMNS].astype({"observation_index": "int64", "arc_id": "int64",
                                                  "cand_rank": "int64", "n_candidates": "int64"})
    return obs_tab, c.reset_index(drop=True)
