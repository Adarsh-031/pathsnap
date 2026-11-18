import pandas as pd
import pytest

from pathsnap.data.reader import Trip
from pathsnap.reconstruction.road_graph import RoadGraph, RoadPath


@pytest.fixture
def graph():
    nodes = pd.DataFrame(
        [(10, 8.60, 49.90), (20, 8.61, 49.90), (30, 8.62, 49.90),
         (40, 8.61, 49.92), (50, 8.63, 49.90)],
        columns=["node_id", "lon", "lat"],
    )
    arcs = pd.DataFrame(
        [(15, 10, 20), (16, 10, 20), (24, 20, 30), (25, 20, 20),
         (30, 10, 40), (31, 40, 30)],
        columns=["arc_id", "from_node", "to_node"],
    )
    observations = pd.DataFrame({"lon": [8.60], "lat": [49.90]})
    return RoadGraph(Trip("00000001", "train", observations, nodes, arcs))


def test_shortest_path_preserves_direction_and_parallel_arcs(graph):
    assert graph.outgoing[10] == [15, 16, 30]
    assert graph.arcs[15] == graph.arcs[16]
    assert graph.node_coordinates[10] == (8.60, 49.90)
    path = graph.shortest_path(10, 30, max_distance_m=10_000)
    assert path is not None
    assert path.arc_ids == (15, 24)
    assert path.distance_m == pytest.approx(1_437, abs=5)
    assert graph.shortest_path(30, 10, max_distance_m=10_000) is None


def test_shortest_path_checks_limits_and_disconnected_nodes(graph):
    path = graph.shortest_path(10, 30, max_distance_m=1_500)
    assert path is not None
    assert graph.shortest_path(10, 30, max_distance_m=path.distance_m) == path
    assert graph.shortest_path(10, 30, max_distance_m=path.distance_m - 1) is None
    assert graph.shortest_path(10, 50, max_distance_m=10_000) is None
    assert graph.shortest_path(20, 20, max_distance_m=0) == RoadPath((), 0.0)
    with pytest.raises(KeyError):
        graph.shortest_path(99, 99, max_distance_m=0)
    with pytest.raises(ValueError):
        graph.shortest_path(10, 30, max_distance_m=-1)
