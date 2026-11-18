import numpy as np
import pandas as pd
import pytest

from pathsnap.data.reader import Trip
from pathsnap.reconstruction import ObservationMatch, decode_route, reconstruct_route
from pathsnap.reconstruction.road_graph import RoadGraph


@pytest.fixture
def trip():
    return Trip(
        "synthetic",
        "train",
        pd.DataFrame({
            "observation_index": [0, 1, 2],
            "lon": [9.00005, 9.00008, 9.00025],
            "lat": [49.0, 49.0, 49.0],
            "t": [0.0, 1.0, 2.0],
        }),
        pd.DataFrame({
            "node_id": range(6),
            "lon": [9.0, 9.0001, 9.0002, 9.0003, 9.0, 9.0001],
            "lat": [49.0, 49.0, 49.0, 49.0, 49.001, 49.001],
        }),
        pd.DataFrame({
            "arc_id": range(4), "from_node": [0, 1, 2, 4], "to_node": [1, 2, 3, 5],
        }),
        route=np.array([3]),
    )


def _observations(positions, gaps):
    return pd.DataFrame({
        "observation_index": range(len(positions)), "x_m": positions,
        "y_m": 0.0, "gap_flag": gaps,
    })


def _candidates(rows):
    return pd.DataFrame(rows, columns=["observation_index", "arc_id", "proj_frac", "cand_rank"])


def test_viterbi_uses_future_observations_and_inserts_connecting_arcs(trip):
    observations = _observations([0, 2, 15], [0, 0, 0])
    candidates = _candidates([
        (0, 3, 0.5, 0), (0, 0, 0.5, 1),
        (1, 3, 0.8, 0), (1, 0, 0.8, 1),
        (2, 2, 0.5, 0),
    ])
    result = decode_route(RoadGraph(trip), observations, candidates, [0.99, 0.1, 0.99, 0.1, 0.9])

    assert [match.arc_id for match in result.matches] == [0, 0, 2]
    assert result.sections == ((0, 1, 2),)


def test_missing_candidates_gaps_and_unreachable_transitions_split_route(trip):
    observations = _observations([0, 2, 4, 6, 8], [0, 0, 0, 1, 0])
    candidates = _candidates([(0, 0, 0.5, 0), (2, 2, 0.1, 0), (3, 2, 0.2, 0), (4, 0, 0.2, 0)])
    result = decode_route(RoadGraph(trip), observations, candidates, [1, 1, 1, 1])

    assert result.matches == (
        ObservationMatch(0, 0, 0), ObservationMatch(1, None, None), ObservationMatch(2, 2, 1),
        ObservationMatch(3, 2, 2), ObservationMatch(4, 0, 3),
    )
    assert result.sections == ((0,), (2,), (2,), (0,))


def test_backward_same_arc_can_follow_directed_loop(trip):
    trip.arcs.loc[4] = [4, 1, 0]
    observations = _observations([0, 18], [0, 0])
    # Extend the first arc to make the backwards movement exceed the jitter allowance.
    trip.nodes.loc[1, "lon"] = 9.00025
    candidates = _candidates([(0, 0, 0.9, 0), (1, 0, 0.1, 0)])
    result = decode_route(RoadGraph(trip), observations, candidates, [1, 1])

    assert result.sections == ((0, 4, 0),)


def test_reconstruction_uses_only_inference_data(trip):
    expected = reconstruct_route(trip.without_route())

    assert reconstruct_route(trip) == expected
    assert expected.sections == ((0, 1, 2),)


@pytest.mark.parametrize("probabilities", [[0.5, 0.5], [np.nan], [1.1]])
def test_rejects_invalid_candidate_probabilities(trip, probabilities):
    with pytest.raises(ValueError, match="one finite probability"):
        decode_route(
            RoadGraph(trip), _observations([0], [0]), _candidates([(0, 0, 0.5, 0)]), probabilities,
        )
