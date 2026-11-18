"""Run the candidate/feature pipeline (and, for dev partitions, derived labels) on one trip.

Usage:
    uv run python -m pathsnap.features.run_trip --root "E:\\pathsnap_data\\Student folder" --trip 00000001
Writes to reports/eda/trip_<id>/:
    observations.csv, candidates.csv              (inference-safe; built without .route)
    labels.csv, coverage.json, validation.json    (reference-derived; dev only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathsnap.data.labels import candidate_coverage, derive_observation_labels, label_summary
from pathsnap.data.reader import load_trip
from pathsnap.data.validate import validate_trip
from .candidates import FEATURE_SCHEMA_VERSION, CandidateConfig, build_candidates


def _jsonable(d: dict) -> dict:
    return {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v) for k, v in d.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--trip", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--radius-m", type=float, default=CandidateConfig.radius_m)
    ap.add_argument("--max-candidates", type=int, default=CandidateConfig.max_candidates)
    a = ap.parse_args(argv)
    out = a.out or Path("reports/eda") / f"trip_{a.trip}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = CandidateConfig(radius_m=a.radius_m, max_candidates=a.max_candidates)

    # 1) inference-safe part: load WITHOUT the route
    trip_inf = load_trip(a.root, a.trip, include_route=False)
    obs, cand = build_candidates(trip_inf, cfg)
    obs.to_csv(out / "observations.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    print(f"[{a.trip}] partition={trip_inf.partition} obs={len(obs)} nodes={len(trip_inf.nodes)} "
          f"arcs={len(trip_inf.arcs)} candidate_rows={len(cand)} "
          f"obs_without_candidates={int((obs['n_candidates'] == 0).sum())} "
          f"feature_schema={FEATURE_SCHEMA_VERSION}")

    # 2) reference part (dev partitions only)
    trip_ref = load_trip(a.root, a.trip, include_route=True)
    (out / "validation.json").write_text(json.dumps(_jsonable(validate_trip(trip_ref)), indent=2))
    if trip_ref.route is None:
        print("No .route file: skipping derived labels.")
        return
    labels = derive_observation_labels(trip_ref)
    labels.to_csv(out / "labels.csv", index=False)
    summ = label_summary(trip_ref, labels)
    cov = candidate_coverage(cand, labels)
    (out / "coverage.json").write_text(json.dumps({"config": cfg.__dict__, "labels": summ,
                                                   "coverage": cov}, indent=2))
    print("label summary:", summ)
    print("candidate coverage:", cov)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
