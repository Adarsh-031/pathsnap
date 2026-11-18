"""Raw PathSnap file reader.

File formats (from "Dataset Exploration.pdf"; verify against the real files):
    <id>.track  3 whitespace columns, no header: longitude, latitude, timestamp_s
    <id>.nodes  2 whitespace columns, no header: longitude, latitude   (row index = node_id)
    <id>.arcs   2 integer columns, no header:    from_node_id, to_node_id (row index = arc_id)
    <id>.route  1 integer column, no header:     arc_id (chronological reference route)

Node and arc IDs are local to each trip's road network.

IMPORTANT: .track rows and .route rows are NOT aligned row by row.
Inference code must call ``load_trip(..., include_route=False)``; the route is
reference information only (see ``pathsnap.data.labels``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["lon", "lat", "t"]
NODE_COLUMNS = ["lon", "lat"]
ARC_COLUMNS = ["from_node", "to_node"]
EXTENSIONS = ("track", "nodes", "arcs", "route")


@dataclass
class Trip:
    """One trip: GPS observations plus its local road network.

    observations: columns observation_index (0..N-1, file order), lon, lat, t
    nodes:        columns node_id, lon, lat
    arcs:         columns arc_id, from_node, to_node
    route:        reference arc_id sequence, or None (never used for inference)
    """

    trip_id: str
    partition: str
    observations: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: np.ndarray | None = None
    source_files: dict[str, Path] = field(default_factory=dict)

    def without_route(self) -> "Trip":
        """Copy safe to hand to the prediction pipeline."""
        return Trip(self.trip_id, self.partition, self.observations, self.nodes,
                    self.arcs, None, {k: v for k, v in self.source_files.items() if k != "route"})


def _read_table(path: Path, names: list[str], dtype: str | dict) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype,
                     engine="python", comment=None)
    if df.shape[1] != len(names):
        raise ValueError(f"{path}: expected {len(names)} columns, got {df.shape[1]}")
    return df


def read_track(path: Path) -> pd.DataFrame:
    df = _read_table(path, TRACK_COLUMNS, "float64")
    df.insert(0, "observation_index", np.arange(len(df), dtype=np.int64))
    return df


def read_nodes(path: Path) -> pd.DataFrame:
    df = _read_table(path, NODE_COLUMNS, "float64")
    df.insert(0, "node_id", np.arange(len(df), dtype=np.int64))
    return df


def read_arcs(path: Path) -> pd.DataFrame:
    df = _read_table(path, ARC_COLUMNS, "int64")
    df.insert(0, "arc_id", np.arange(len(df), dtype=np.int64))
    return df


def read_route(path: Path) -> np.ndarray:
    df = _read_table(path, ["arc_id"], "int64")
    return df["arc_id"].to_numpy()


def _partition_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0].lower() if len(rel.parts) > 1 else "unknown"


def discover_trips(root: str | Path, *, partition: str | None = None) -> dict[str, dict]:
    """Find trips under ``root`` by grouping files that share a stem in one directory.

    Returns {trip_id: {"partition": str, "files": {ext: Path}}}.
    Partition = first directory under root (lower-cased). Pass its directory name
    to scan just that partition. Nothing about counts is assumed.
    """
    root = Path(root)
    found: dict[str, dict] = {}
    search_root = root if partition is None else root / partition
    for path in sorted(search_root.rglob("*")):
        ext = path.suffix.lstrip(".").lower()
        if not path.is_file() or ext not in EXTENSIONS:
            continue
        trip_id = path.stem
        entry = found.setdefault(trip_id, {"partition": _partition_of(path, root), "files": {}})
        if ext in entry["files"]:
            raise ValueError(f"Duplicate {ext} file for trip {trip_id}: "
                             f"{entry['files'][ext]} and {path}")
        entry["files"][ext] = path
    return found


def load_trip(root: str | Path, trip_id: str, include_route: bool = True) -> Trip:
    """Load one trip. Set include_route=False on any inference path."""
    trips = discover_trips(root)
    if trip_id not in trips:
        raise KeyError(f"Trip {trip_id!r} not found under {root}")
    info = trips[trip_id]
    return load_trip_files(trip_id, info["partition"], info["files"], include_route=include_route)


def load_trip_files(
    trip_id: str, partition: str, files: dict[str, Path], *, include_route: bool,
) -> Trip:
    """Read an already discovered trip, optionally including its scoring reference."""
    missing = [e for e in ("track", "nodes", "arcs") if e not in files]
    if missing:
        raise FileNotFoundError(f"Trip {trip_id} missing required files: {missing}")
    route = read_route(files["route"]) if include_route and "route" in files else None
    kept = dict(files) if include_route else {k: v for k, v in files.items() if k != "route"}
    return Trip(trip_id, partition, read_track(files["track"]),
                read_nodes(files["nodes"]), read_arcs(files["arcs"]), route, kept)
