"""Decode directed road candidates with a hidden Markov model."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np
import pandas as pd

from pathsnap.data.reader import Trip
from pathsnap.features.candidates import FEATURE_COLUMNS, CandidateConfig, build_candidates
from .road_graph import RoadGraph, RoadPath


@dataclass(frozen=True)
class HMMConfig:
    gps_sigma_m: float = 10.0
    transition_beta_m: float = 20.0
    max_detour_m: float = 100.0
    backward_tolerance_m: float = 10.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.gps_sigma_m == 0 or self.transition_beta_m == 0:
            raise ValueError("gps_sigma_m and transition_beta_m must be positive")


@dataclass(frozen=True)
class ObservationMatch:
    observation_index: int
    arc_id: int | None
    section_index: int | None


@dataclass(frozen=True)
class ReconstructedRoute:
    matches: tuple[ObservationMatch, ...]
    sections: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _Candidate:
    arc_id: int
    fraction: float
    emission: float


@dataclass(frozen=True)
class _State:
    candidate: _Candidate
    score: float
    previous: _State | None = None
    incoming_arcs: tuple[int, ...] = ()


def _advance(
    graph: RoadGraph,
    previous: list[_State],
    candidates: list[_Candidate],
    gps_distance: float,
    config: HMMConfig,
) -> list[_State]:
    # The limit applies to travel between projections, including partial arcs.
    limit = gps_distance + config.max_detour_m
    paths: dict[tuple[int, int], RoadPath | None] = {}
    current = []
    for candidate in candidates:
        arc = graph.arcs[candidate.arc_id]
        best = None
        for state in previous:
            prior = state.candidate
            prior_arc = graph.arcs[prior.arc_id]
            progress = (candidate.fraction - prior.fraction) * arc.length_m
            if candidate.arc_id == prior.arc_id and progress >= -config.backward_tolerance_m:
                distance = max(0.0, progress)
                incoming = ()
            else:
                partial_distance = (
                    (1 - prior.fraction) * prior_arc.length_m + candidate.fraction * arc.length_m
                )
                if partial_distance > limit:
                    continue
                endpoints = (prior_arc.to_node, arc.from_node)
                if endpoints not in paths:
                    paths[endpoints] = graph.shortest_path(*endpoints, max_distance_m=limit)
                path = paths[endpoints]
                if path is None:
                    continue
                distance = partial_distance + path.distance_m
                incoming = (*path.arc_ids, candidate.arc_id)

            if distance <= limit:
                transition = -abs(distance - gps_distance) / config.transition_beta_m
                score = state.score + transition + candidate.emission
                if best is None or score > best.score:
                    best = _State(candidate, score, state, incoming)
        if best is not None:
            current.append(best)
    return current


def _finish_section(
    matches: list[ObservationMatch],
    sections: list[tuple[int, ...]],
    indices: list[int],
    states: list[_State],
) -> None:
    if not states:
        return
    state = max(states, key=lambda state: state.score)
    chain = [state]
    while state.previous is not None:
        state = state.previous
        chain.append(state)
    chain.reverse()

    arc_ids = [chain[0].candidate.arc_id]
    for index, state in zip(indices, chain, strict=True):
        matches.append(ObservationMatch(index, state.candidate.arc_id, len(sections)))
        arc_ids.extend(state.incoming_arcs)
    sections.append(tuple(arc_ids))


def decode_route(
    graph: RoadGraph,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    probabilities: np.ndarray | None = None,
    *,
    config: HMMConfig = HMMConfig(),
) -> ReconstructedRoute:
    """Decode one trip's candidate table in observation order.

    Supply positive-class probabilities in candidate row order. When omitted,
    use a Gaussian distance emission. Each section contains the traversed road
    arc IDs, including any intermediate arcs inferred between observations.
    """
    if probabilities is None:
        emissions = -0.5 * np.square(candidates["dist_m"].to_numpy() / config.gps_sigma_m)
    else:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(candidates),) or not (
            np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all()
        ):
            raise ValueError("Expected one finite probability in [0, 1] per candidate row")
        emissions = np.log(np.maximum(probabilities, np.finfo(float).tiny))

    by_observation: dict[int, list[_Candidate]] = {}
    ordered = candidates.assign(emission=emissions).sort_values(["cand_rank", "arc_id"])
    columns = ["observation_index", "arc_id", "proj_frac", "emission"]
    for index, arc_id, fraction, emission in ordered[columns].itertuples(index=False, name=None):
        by_observation.setdefault(index, []).append(_Candidate(arc_id, fraction, emission))

    matches: list[ObservationMatch] = []
    sections: list[tuple[int, ...]] = []
    indices: list[int] = []
    states: list[_State] = []
    prior_x, prior_y = 0.0, 0.0
    columns = ["observation_index", "x_m", "y_m", "gap_flag"]
    for index, x, y, gap in observations.sort_values("observation_index")[columns].itertuples(
        index=False, name=None,
    ):
        options = by_observation.get(index, [])
        advanced = []
        if states and options and not gap:
            advanced = _advance(graph, states, options, hypot(x - prior_x, y - prior_y), config)
        if not advanced:
            _finish_section(matches, sections, indices, states)
            indices = []
            advanced = [_State(candidate, candidate.emission) for candidate in options]

        states = advanced
        if states:
            indices.append(index)
        else:
            matches.append(ObservationMatch(index, None, None))
        prior_x, prior_y = x, y

    _finish_section(matches, sections, indices, states)
    return ReconstructedRoute(tuple(matches), tuple(sections))


def candidate_probabilities(classifier, candidates: pd.DataFrame) -> np.ndarray:
    """Score candidates with a fitted binary classifier using the feature contract."""
    positive_class, = np.flatnonzero(np.asarray(classifier.classes_) == 1)
    if candidates.empty:
        return np.empty(0)
    return classifier.predict_proba(candidates[FEATURE_COLUMNS])[:, positive_class]


def reconstruct_route(
    trip: Trip,
    *,
    classifier=None,
    candidate_config: CandidateConfig = CandidateConfig(),
    config: HMMConfig = HMMConfig(),
) -> ReconstructedRoute:
    """Build and decode candidates using only the GPS track and road network."""
    if trip.observations.empty:
        return ReconstructedRoute((), ())

    observations, candidates = build_candidates(trip, candidate_config)
    probabilities = None if classifier is None else candidate_probabilities(classifier, candidates)
    return decode_route(RoadGraph(trip), observations, candidates, probabilities, config=config)
