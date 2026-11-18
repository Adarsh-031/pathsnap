"""Candidate coverage (candidate miss rate) over all development trips.

Usage:
    uv run python -m pathsnap.features.coverage --root "<data root>" --out reports/eda
Writes <out>/candidate_coverage.csv (one row per trip) and prints partition totals.
Uses derived labels, so this is a development diagnostic only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pathsnap.data.labels import candidate_coverage, derive_observation_labels
from pathsnap.data.reader import discover_trips, load_trip
from .candidates import CandidateConfig, build_candidates


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("reports/eda"), type=Path)
    ap.add_argument("--radius-m", type=float, default=CandidateConfig.radius_m)
    ap.add_argument("--max-candidates", type=int, default=CandidateConfig.max_candidates)
    a = ap.parse_args(argv)
    cfg = CandidateConfig(radius_m=a.radius_m, max_candidates=a.max_candidates)
    rows = []
    for trip_id, info in sorted(discover_trips(a.root).items()):
        if "route" not in info["files"]:
            continue
        trip = load_trip(a.root, trip_id)
        obs, cand = build_candidates(trip.without_route(), cfg)
        lab = derive_observation_labels(trip)
        rows.append({"trip_id": trip_id, "partition": trip.partition,
                     "obs_without_candidates": int((obs["n_candidates"] == 0).sum()),
                     "candidate_rows": len(cand), **candidate_coverage(cand, lab)})
        print(trip_id, rows[-1]["miss_rate_ok_only"], flush=True)
    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "candidate_coverage.csv", index=False)
    g = df.groupby("partition")[["n_labelled", "miss_count_all", "miss_count_ok_only",
                                  "obs_without_candidates", "candidate_rows"]].sum()
    print(f"config: radius_m={cfg.radius_m} max_candidates={cfg.max_candidates}")
    print(g.to_string())


if __name__ == "__main__":
    main()
