import argparse
import re
from pathlib import Path
from typing import Callable, TypeVar

from .maps import development_map

Value = TypeVar("Value", int, float)


def read_rows(path: Path, columns: int, cast: Callable[[str], Value]) -> list[tuple[Value, ...]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for row_number, line in enumerate(source, start=1):
            fields = line.split()
            if len(fields) != columns:
                raise ValueError(f"{path.name}:{row_number}: expected {columns} fields")
            rows.append(tuple(map(cast, fields)))
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a development Folium map for a training trip.")
    parser.add_argument("--data-root", required=True, type=Path, help="Directory containing train/")
    parser.add_argument("--trip-id", default="00000001")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{8}", args.trip_id):
        parser.error("--trip-id must have eight digits")

    prefix = args.data_root / "train" / args.trip_id / args.trip_id
    try:
        gps = [(lon, lat) for lon, lat, _ in read_rows(prefix.with_suffix(".track"), 3, float)]
        nodes = [(lon, lat) for lon, lat in read_rows(prefix.with_suffix(".nodes"), 2, float)]
        arcs = [(start, end) for start, end in read_rows(prefix.with_suffix(".arcs"), 2, int)]
        route = [arc_id for (arc_id,) in read_rows(prefix.with_suffix(".route"), 1, int)]
        if any(not (0 <= start < len(nodes) and 0 <= end < len(nodes)) for start, end in arcs):
            raise ValueError("Road arc contains an out-of-range node ID")
        if any(not 0 <= arc_id < len(arcs) for arc_id in route):
            raise ValueError("Reference route contains an out-of-range arc ID")
        result, visible_roads = development_map(args.trip_id, gps, nodes, arcs, route)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(args.output))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Could not generate development map: {error}\n")
    print(
        f"{args.output}: {len(gps):,} GPS points, {visible_roads:,} displayed road segments, "
        f"{len(route):,} reference arcs"
    )


if __name__ == "__main__":
    main()
