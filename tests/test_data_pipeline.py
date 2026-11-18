"""Unit tests on a tiny SYNTHETIC trip (not real PathSnap data).

Road: straight west->east chain of 6 nodes 100 m apart, two-way (forward + reverse arcs),
plus a parallel road 40 m north. Route = forward arcs 0..4. Track drives along the main
road at 10 m/s with small offsets, so there are more observations than route entries.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathsnap.data.labels import (attach_training_target, candidate_coverage,
                                  derive_observation_labels)
from pathsnap.data.reader import discover_trips, load_trip
from pathsnap.data.splits import trip_group_folds
from pathsnap.data.validate import validate_trip
from pathsnap.features.candidates import FEATURE_COLUMNS, KEY_COLUMNS, build_candidates
from pathsnap.features.geometry import Projector

LON0, LAT0 = 8.6, 49.9
M_PER_DEG_LAT = 111_200.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(LAT0))


def _ll(x, y):
    return LON0 + x / M_PER_DEG_LON, LAT0 + y / M_PER_DEG_LAT


def write_trip(root: Path, partition="train", trip_id="00000007", with_route=True, gap=False):
    d = root / partition / trip_id
    d.mkdir(parents=True)
    nodes = [_ll(100 * k, 0) for k in range(6)] + [_ll(100 * k, 40) for k in range(6)]
    arcs = [(k, k + 1) for k in range(5)]                  # 0..4 forward main road
    arcs += [(k + 1, k) for k in range(5)]                 # 5..9 reverse main road
    arcs += [(6 + k, 7 + k) for k in range(5)]             # 10..14 parallel road
    t = np.arange(50, dtype=float)
    if gap:
        t[25:] += 20
    xs = 10 * np.arange(50) + 2.0
    ys = np.where(np.arange(50) % 2, 3.0, -3.0)
    track = [(*_ll(x, y), tt) for x, y, tt in zip(xs, ys, t)]
    np.savetxt(d / f"{trip_id}.nodes", nodes, fmt="%.8f")
    np.savetxt(d / f"{trip_id}.arcs", arcs, fmt="%d")
    np.savetxt(d / f"{trip_id}.track", track, fmt="%.8f")
    if with_route:
        np.savetxt(d / f"{trip_id}.route", [0, 1, 2, 3, 4], fmt="%d")
    return root


def test_reader_ids_and_partition(tmp_path):
    root = write_trip(tmp_path)
    trip = load_trip(root, "00000007")
    assert trip.partition == "train"
    assert list(trip.nodes["node_id"]) == list(range(12))
    assert list(trip.arcs["arc_id"]) == list(range(15))
    assert list(trip.observations["observation_index"]) == list(range(50))
    assert list(trip.route) == [0, 1, 2, 3, 4]
    assert load_trip(root, "00000007", include_route=False).route is None
    assert "route" not in trip.without_route().source_files


def test_discover_multiple_partitions(tmp_path):
    write_trip(tmp_path, "train", "00000001")
    write_trip(tmp_path, "student_test", "00000002", with_route=False)
    found = discover_trips(tmp_path)
    assert found["00000001"]["partition"] == "train"
    assert found["00000002"]["partition"] == "student_test"
    assert "route" not in found["00000002"]["files"]


def test_validation_flags(tmp_path):
    trip = load_trip(write_trip(tmp_path, gap=True), "00000007")
    v = validate_trip(trip)
    assert v["track_n_gaps_gt_5s"] == 1
    assert v["arcs_n_invalid_node_ref"] == 0
    assert v["route_n_discontinuities"] == 0
    assert v["arcs_n_with_reverse"] == 10


def test_projection_distance_in_metres():
    p = Projector(LON0, LAT0)
    x, y = p.to_xy([LON0, LON0], [LAT0, LAT0 + 0.001])
    assert abs(np.hypot(x[1] - x[0], y[1] - y[0]) - 111.2) < 0.5


def test_candidates_schema_and_no_route(tmp_path):
    trip = load_trip(write_trip(tmp_path), "00000007")
    obs, cand = build_candidates(trip)
    assert list(cand.columns) == KEY_COLUMNS + FEATURE_COLUMNS
    assert len(obs) == 50 and (obs["n_candidates"] > 0).all()
    assert cand.groupby("observation_index")["cand_rank"].min().eq(0).all()
    # forward arc must beat its reverse twin on heading
    o10 = cand[cand.observation_index == 10].set_index("arc_id")
    assert o10.loc[1, "heading_diff_deg"] < 10 and o10.loc[6, "heading_diff_deg"] > 170


def test_far_observation_kept_without_candidates(tmp_path):
    root = write_trip(tmp_path)
    p = root / "train" / "00000007" / "00000007.track"
    rows = np.loadtxt(p)
    rows[5, :2] = _ll(50, 500)  # 500 m off the network
    np.savetxt(p, rows, fmt="%.8f")
    obs, cand = build_candidates(load_trip(root, "00000007"))
    assert obs.loc[5, "n_candidates"] == 0 and obs.loc[5, "has_candidates"] == 0
    assert 5 not in set(cand["observation_index"])


def test_labels_monotone_and_correct(tmp_path):
    trip = load_trip(write_trip(tmp_path), "00000007")
    lab = derive_observation_labels(trip)
    assert len(lab) == 50 != len(trip.route)
    assert (np.diff(lab["route_position"]) >= 0).all()
    expected = np.minimum((10 * np.arange(50) + 2) // 100, 4)
    assert (lab["label_arc_id"].to_numpy() == expected).mean() > 0.95
    assert set(lab["label_status"]) <= {"ok", "ambiguous", "far"}
    assert lab.loc[10, "near_route_boundary"] == 1
    assert lab.loc[10, "label_status"] == "ambiguous"
    assert lab.loc[5, "label_status"] == "ok"
    _, cand = build_candidates(trip)
    cov = candidate_coverage(cand, lab)
    assert cov["miss_count_all"] == 0
    y = attach_training_target(cand, lab)
    assert "y" not in cand.columns and y.groupby(cand["observation_index"]).sum().eq(1).all()


def test_labels_flag_nonadjacent_route_competitor(tmp_path):
    trip = load_trip(write_trip(tmp_path), "00000007")
    # Return on the same road after the outbound leg.
    trip = replace(trip, route=np.r_[trip.route, 9:4:-1])
    label = derive_observation_labels(trip).iloc[5]
    assert label["label_arc_id"] == 0
    assert label["near_route_boundary"] == 0
    assert label["label_status"] == "ambiguous"


def test_labels_ignore_repeated_arc_id_and_prioritize_far(tmp_path):
    trip = load_trip(write_trip(tmp_path), "00000007")
    labels = derive_observation_labels(replace(trip, route=np.array([0, 0])))
    assert labels.loc[5, "near_route_boundary"] == 0
    assert labels.loc[5, "label_status"] == "ok"
    assert labels.loc[15, "nearest_nonroute_dist_m"] < labels.loc[15, "label_dist_m"]
    assert labels.loc[15, "label_status"] == "far"


def test_trip_group_folds_no_leak():
    df = pd.DataFrame({"trip_id": np.repeat([f"{i:08d}" for i in range(10)], 5)})
    seen = set()
    for tr, va in trip_group_folds(df, n_splits=5):
        assert not set(df.trip_id.iloc[tr]) & set(df.trip_id.iloc[va])
        seen |= set(df.trip_id.iloc[va])
    assert len(seen) == 10
