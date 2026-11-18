"""Laptop 1 inference API for the PathSnap evaluator UI."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from pathsnap.data.labels import derive_observation_labels
from pathsnap.data.reader import EXTENSIONS, Trip, discover_trips, load_trip_files, read_route
from pathsnap.features.candidates import CandidateConfig, build_candidates
from pathsnap.models.artifact import load_model_artifact
from pathsnap.reconstruction.hmm import ReconstructedRoute, candidate_probabilities, decode_route
from pathsnap.reconstruction.road_graph import RoadGraph

logger = logging.getLogger(__name__)
TRIP_ID_PATTERN = r"^[0-9]{8}$"
TripId = Annotated[str, StringConstraints(strict=True, min_length=8, max_length=8,
                                         pattern=TRIP_ID_PATTERN)]
Position = tuple[float, float]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class TripsResponse(BaseModel):
    trips: list[TripId]


class PredictRequest(BaseModel):
    trip_id: TripId


class LineString(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[Position]


class MultiLineString(BaseModel):
    type: Literal["MultiLineString"] = "MultiLineString"
    coordinates: list[list[Position]]


class Metrics(BaseModel):
    accuracy: float
    matched_points: int
    total_points: int


class Timings(BaseModel):
    inference: float
    total: float


class Prediction(BaseModel):
    predicted_route: Annotated[LineString | MultiLineString, Field(discriminator="type")]
    metrics: Metrics
    timings_ms: Timings


@dataclass(frozen=True)
class InferenceService:
    classifier: object
    candidate_config: CandidateConfig
    partition: str
    trips: dict[str, dict[str, Path]]

    @classmethod
    def from_environment(cls) -> "InferenceService":
        root = os.environ.get("PATHSNAP_DATA_ROOT")
        model = os.environ.get("PATHSNAP_MODEL")
        if not root or not model:
            raise ValueError("Set PATHSNAP_DATA_ROOT and PATHSNAP_MODEL")
        partition = os.environ.get("PATHSNAP_TRIP_PARTITION", "student_test")
        if not partition or Path(partition).name != partition or partition in {".", ".."}:
            raise ValueError("PATHSNAP_TRIP_PARTITION must be a directory name")

        trips = {}
        for trip_id, info in discover_trips(root, partition=partition).items():
            if not re.fullmatch(TRIP_ID_PATTERN, trip_id):
                continue
            files = info["files"]
            missing = set(EXTENSIONS) - files.keys()
            if missing:
                raise ValueError(f"Trip {trip_id} is missing files: {', '.join(sorted(missing))}")
            trips[trip_id] = files
        if not trips:
            raise ValueError(f"No trips found in {Path(root) / partition}")

        classifier, candidate_config = load_model_artifact(Path(model))
        if (not callable(getattr(classifier, "predict_proba", None))
                or sorted(getattr(classifier, "classes_", [])) != [0, 1]):
            raise ValueError("Expected a fitted binary classifier with classes 0 and 1")
        return cls(classifier, candidate_config, partition, trips)

    def predict(self, trip_id: str) -> Prediction:
        started = perf_counter()
        files = self.trips.get(trip_id)
        if files is None:
            raise HTTPException(404, "Trip is outside the active list")

        try:
            trip = load_trip_files(trip_id, self.partition, files, include_route=False)
            if trip.observations.empty:
                raise ValueError("Trip has an empty GPS track")
            observations, candidates = build_candidates(trip, self.candidate_config)
            graph = RoadGraph(trip)
        except OSError:
            logger.exception("Input files are unavailable for trip %s", trip_id)
            raise HTTPException(503, "Trip files are unavailable") from None
        except (ValueError, IndexError, KeyError):
            logger.exception("Invalid input data for trip %s", trip_id)
            raise HTTPException(422, "Trip has invalid GPS or road-network data") from None

        inference_started = perf_counter()
        try:
            probabilities = candidate_probabilities(self.classifier, candidates)
            result = decode_route(graph, observations, candidates, probabilities)
        except Exception:
            logger.exception("Inference failed for trip %s", trip_id)
            raise HTTPException(503, "Inference is unavailable") from None
        inference_ms = (perf_counter() - inference_started) * 1000

        geometry = route_geometry(graph, result)
        metrics = score_route(trip, files["route"], result)
        response = Prediction(
            predicted_route=geometry,
            metrics=metrics,
            timings_ms=Timings(inference=inference_ms, total=0),
        )
        response.timings_ms.total = (perf_counter() - started) * 1000
        return response


def route_geometry(graph: RoadGraph, result: ReconstructedRoute) -> LineString | MultiLineString:
    lines = []
    for section in result.sections:
        positions = []
        for arc_id in section:
            arc = graph.arcs[arc_id]
            for node_id in (arc.from_node, arc.to_node):
                lon, lat = graph.node_coordinates[node_id]
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    raise HTTPException(422, "Trip has invalid route coordinates")
                position = (float(lon), float(lat))
                if not positions or position != positions[-1]:
                    positions.append(position)
        if len(positions) >= 2:
            lines.append(positions)

    if not lines:
        raise HTTPException(422, "Trip has no drawable route")
    if len(lines) == 1:
        return LineString(coordinates=lines[0])
    return MultiLineString(coordinates=lines)


def score_route(trip: Trip, reference_file: Path, result: ReconstructedRoute) -> Metrics:
    try:
        trip_with_reference = replace(trip, route=read_route(reference_file))
        labels = derive_observation_labels(trip_with_reference)
    except OSError:
        logger.exception("Scoring reference is unavailable for trip %s", trip.trip_id)
        raise HTTPException(503, "Scoring reference is unavailable") from None
    except ValueError:
        logger.exception("Invalid scoring reference for trip %s", trip.trip_id)
        raise HTTPException(422, "Trip has an invalid scoring reference") from None

    scorable = labels.loc[labels["label_status"] == "ok", ["observation_index", "label_arc_id"]]
    if scorable.empty:
        raise HTTPException(422, "Trip has no scorable observations")
    predictions = {match.observation_index: match.arc_id for match in result.matches}
    matched = sum(predictions.get(index) == arc_id
                  for index, arc_id in scorable.itertuples(index=False, name=None))
    return Metrics(accuracy=matched / len(scorable), matched_points=matched,
                   total_points=len(scorable))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = None
    try:
        app.state.inference_service = InferenceService.from_environment()
    except Exception:
        logger.exception("Could not initialize PathSnap inference")
    yield


def ready_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise HTTPException(503, "Inference is unavailable")
    return service


ReadyService = Annotated[InferenceService, Depends(ready_service)]
app = FastAPI(title="PathSnap Inference API", lifespan=lifespan)


@app.get(
    "/health",
    dependencies=[Depends(ready_service)],
    responses={503: {"description": "Inference is unavailable"}},
)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/trips", responses={503: {"description": "Inference is unavailable"}})
def trips(service: ReadyService) -> TripsResponse:
    return TripsResponse(trips=sorted(service.trips))


@app.post(
    "/predict",
    responses={
        404: {"description": "Trip is outside the active list"},
        503: {"description": "Inference, trip files, or scoring reference is unavailable"},
    },
)
def predict(payload: PredictRequest, service: ReadyService) -> Prediction:
    return service.predict(payload.trip_id)
