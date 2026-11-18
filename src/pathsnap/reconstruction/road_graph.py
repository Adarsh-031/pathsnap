"""Trip-local directed road graph for map matching."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot, inf, isfinite

from pathsnap.data.reader import Trip
from pathsnap.features.geometry import Projector


@dataclass(frozen=True)
class RoadArc:
    from_node: int
    to_node: int
    length_m: float


@dataclass(frozen=True)
class RoadPath:
    arc_ids: tuple[int, ...]
    distance_m: float


class RoadGraph:
    """Index the road network by its original node and directed arc IDs."""

    def __init__(self, trip: Trip):
        nodes = trip.nodes
        lon = nodes["lon"].to_numpy()
        lat = nodes["lat"].to_numpy()
        node_ids = nodes["node_id"].to_list()
        self.node_coordinates = dict(zip(node_ids, zip(lon, lat), strict=True))

        observations = trip.observations
        projector = Projector.for_lonlat(
            observations["lon"].to_numpy(), observations["lat"].to_numpy(),
        )
        x, y = projector.to_xy(lon, lat)
        projected = dict(zip(node_ids, zip(x, y), strict=True))

        self.arcs: dict[int, RoadArc] = {}
        self.outgoing: dict[int, list[int]] = {node_id: [] for node_id in node_ids}
        for arc_id, start, end in trip.arcs[["arc_id", "from_node", "to_node"]].itertuples(
            index=False, name=None,
        ):
            start_x, start_y = projected[start]
            end_x, end_y = projected[end]
            self.arcs[arc_id] = RoadArc(start, end, hypot(end_x - start_x, end_y - start_y))
            self.outgoing[start].append(arc_id)

    def shortest_path(
        self, start_node: int, end_node: int, *, max_distance_m: float,
    ) -> RoadPath | None:
        """Find the shortest directed path within the distance limit, in metres."""
        if not isfinite(max_distance_m) or max_distance_m < 0:
            raise ValueError("max_distance_m must be finite and nonnegative")
        for node_id in (start_node, end_node):
            if node_id not in self.outgoing:
                raise KeyError(node_id)

        distances = {start_node: 0.0}
        previous_arc: dict[int, int] = {}
        queue = [(0.0, start_node)]
        while queue:
            distance, node = heappop(queue)
            if distance > distances[node]:
                continue
            if node == end_node:
                arc_ids = []
                while node != start_node:
                    arc_id = previous_arc[node]
                    arc_ids.append(arc_id)
                    node = self.arcs[arc_id].from_node
                return RoadPath(tuple(reversed(arc_ids)), distance)

            for arc_id in self.outgoing[node]:
                arc = self.arcs[arc_id]
                candidate_distance = distance + arc.length_m
                if (
                    candidate_distance <= max_distance_m
                    and candidate_distance < distances.get(arc.to_node, inf)
                ):
                    distances[arc.to_node] = candidate_distance
                    previous_arc[arc.to_node] = arc_id
                    heappush(queue, (candidate_distance, arc.to_node))
        return None
