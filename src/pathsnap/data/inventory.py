"""Inventory + validation of every trip under a data root.

Usage:
    uv run python -m pathsnap.data.inventory --root "E:\\pathsnap_data\\Student folder" --out reports/eda
Writes:
    <out>/inventory.csv         one row per trip, validation columns from validate_trip
    <out>/label_summary.csv     derived-label quality per trip (only trips that have .route)
    <out>/inventory_summary.txt short human-readable summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .labels import derive_observation_labels, label_summary
from .reader import discover_trips, load_trip
from .validate import validate_trip


def run(root: Path, out: Path, with_labels: bool = True) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    trips = discover_trips(root)
    rows, lab_rows = [], []
    for trip_id, info in sorted(trips.items()):
        files = info["files"]
        row = {"trip_id": trip_id, "partition": info["partition"],
               **{f"has_{e}": e in files for e in ("track", "nodes", "arcs", "route")}}
        try:
            trip = load_trip(root, trip_id)
            row.update(validate_trip(trip))
            if with_labels and trip.route is not None:
                lab_rows.append(label_summary(trip, derive_observation_labels(trip)))
        except Exception as exc:  # keep going; record the failure
            row["load_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    inv = pd.DataFrame(rows)
    inv.to_csv(out / "inventory.csv", index=False)
    if lab_rows:
        pd.DataFrame(lab_rows).to_csv(out / "label_summary.csv", index=False)

    lines = [f"Data root: {root}", f"Trips found: {len(inv)}"]
    if len(inv):
        lines.append("Trips per partition:\n" + inv["partition"].value_counts().to_string())
        for col in ("n_obs", "n_nodes", "n_arcs", "n_route"):
            if col in inv:
                lines.append(f"{col}: " + inv[col].describe().round(1).to_string().replace("\n", "; "))
        flags = ["track_n_nonfinite", "track_n_out_of_range", "track_n_nonincreasing",
                 "track_n_gaps_gt_5s", "arcs_n_invalid_node_ref", "route_n_invalid_arc_ref",
                 "route_n_discontinuities", "nodes_n_isolated"]
        for f in flags:
            if f in inv:
                bad = inv.loc[inv[f].fillna(0) > 0, "trip_id"].tolist()
                lines.append(f"{f}: {len(bad)} trips with >0 {bad[:10]}{' ...' if len(bad) > 10 else ''}")
        if "load_error" in inv:
            lines.append(f"Load errors: {inv['load_error'].notna().sum()}")
    (out / "inventory_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return inv


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("reports/eda"), type=Path)
    ap.add_argument("--no-labels", action="store_true", help="skip derived-label summary")
    a = ap.parse_args(argv)
    run(a.root, a.out, not a.no_labels)


if __name__ == "__main__":
    main()
