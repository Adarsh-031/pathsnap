import argparse
import re
from pathlib import Path

from pathsnap.data import load_trip

from .maps import development_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a development Folium map for a training trip.")
    parser.add_argument("--data-root", required=True, type=Path, help="Directory containing train/")
    parser.add_argument("--trip-id", default="00000001")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{8}", args.trip_id):
        parser.error("--trip-id must have eight digits")

    training_track = args.data_root / "train" / args.trip_id / f"{args.trip_id}.track"
    if not training_track.is_file():
        parser.error(f"Training trip {args.trip_id} not found under {args.data_root / 'train'}")

    try:
        trip = load_trip(args.data_root, args.trip_id, include_route=True)
        if trip.route is None or not len(trip.route):
            raise ValueError("Training trip has no reference route")
        gps = list(trip.observations[["lon", "lat"]].itertuples(index=False, name=None))
        nodes = list(trip.nodes[["lon", "lat"]].itertuples(index=False, name=None))
        arcs = list(trip.arcs[["from_node", "to_node"]].itertuples(index=False, name=None))
        route = trip.route.tolist()
        if any(not (0 <= start < len(nodes) and 0 <= end < len(nodes)) for start, end in arcs):
            raise ValueError("Road arc contains an out-of-range node ID")
        if any(not 0 <= arc_id < len(arcs) for arc_id in route):
            raise ValueError("Reference route contains an out-of-range arc ID")
        result, visible_roads = development_map(args.trip_id, gps, nodes, arcs, route)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(args.output))
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Could not generate development map: {error}\n")
    print(
        f"{args.output}: {len(gps):,} GPS points, {visible_roads:,} displayed road segments, "
        f"{len(route):,} reference arcs"
    )


if __name__ == "__main__":
    main()
