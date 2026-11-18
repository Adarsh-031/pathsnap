# Laptop 1 API handoff

The [FastAPI implementation](../src/pathsnap/api.py) and the Laptop 2 adapter use this contract. The examples use a permitted `student_test` trip ID and synthetic results for illustration. The running server exposes its OpenAPI documentation at `/docs`.

All example responses use HTTP `200` and `application/json`.

## `GET /health`

```json
{"status":"ok"}
```

Return `200` when the server can accept prediction requests. Return `503` while the model or required artifacts are unavailable. Laptop 2 checks the HTTP status.

## `GET /trips`

```json
{"trips":["00000000"]}
```

Return unique eight-digit strings from the server's active trip list. Use student-development trips for rehearsal and the ten reserved trips during authorized evaluation.

## `POST /predict`

Send `Content-Type: application/json` and the selected trip ID as a string so leading zeros are preserved.

```json
{"trip_id":"00000000"}
```

Example response with one route break:

```json
{
  "predicted_route": {
    "type": "MultiLineString",
    "coordinates": [
      [[8.59, 49.84], [8.60, 49.85]],
      [[8.63, 49.87], [8.64, 49.88]]
    ]
  },
  "metrics": {"accuracy": 0.9, "matched_points": 90, "total_points": 100},
  "timings_ms": {"inference": 125, "total": 180}
}
```

- `predicted_route` is GeoJSON geometry in route order with `[longitude, latitude]` pairs in decimal degrees. Use `LineString` with a single array of pairs for a continuous route. With gaps or disconnected joins, use `MultiLineString` with a separate array for each continuous section. Each section needs at least two positions.
- `metrics.accuracy` is the fraction `matched_points / total_points` on a 0–1 scale. `total_points` counts GPS observations with a scorable reference arc (derived labels with `label_status == "ok"`). `matched_points` counts those observations where the final decoded directed arc equals the reference arc. Missing predictions and candidate misses count as incorrect when the observation has a scorable reference. The UI converts the fraction to a percentage.
- `timings_ms.inference` is elapsed classifier scoring plus HMM/Viterbi decoding time, in milliseconds. `timings_ms.total` is elapsed server processing time through response construction. It also includes trip loading, feature generation, scoring against references, and geometry assembly. The client round trip is outside both timers.
- Keep reference labels and reference geometry on Laptop 1. Load them only for server-side scoring after prediction. The response contains predicted geometry and aggregate metrics.

The API returns `404` for a trip outside the active list, `422` for an invalid request or a trip with no drawable route or no scorable observations, and `503` when inference is unavailable. FastAPI can return a `detail` field. The adapter displays the upstream HTTP status for errors.

## LAN rehearsal

Follow the [Laptop 1 startup instructions](ui_deployment.md#laptop-1-fastapi-and-inference) with a fitted model and `student_test` active. Use the confirmed LAN URL to [start the UI on Laptop 2](ui_deployment.md#laptop-2-evaluator-ui). Select a trip, submit it, and verify that Laptop 1 logs the selected trip ID and a successful `POST /predict`. Confirm the decoded route, accuracy, and timings in the UI.
