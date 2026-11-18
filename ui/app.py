import argparse
import json
import math
import re
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from pathsnap.visualization.maps import prediction_map

PAGE = Path(__file__).with_name("index.html")
METRICS = {
    "accuracy": "Accuracy",
    "coverage": "Coverage",
    "route_continuity": "Route continuity",
    "matched_points": "Matched points",
    "total_points": "Total points",
}


class BackendError(Exception):
    pass


def backend_url(value):
    try:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"} or not url.hostname
            or url.username is not None or url.password is not None
            or url.query or url.fragment or url.path not in {"", "/"}
            or url.port == 0
        ):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use an HTTP(S) backend URL with host and optional port.") from error
    return value.rstrip("/")


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9]{8}", value) is not None


def numeric(value):
    return type(value) in {float, int} and math.isfinite(value)


def decode_trips(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("trips"), list):
        raise BackendError("Trip list does not match the provisional API format.")
    trips = payload["trips"]
    if not all(valid_id(trip_id) for trip_id in trips) or len(set(trips)) != len(trips):
        raise BackendError("Backend returned invalid or duplicate trip IDs.")
    return trips


def decode_prediction(payload):
    message = "Prediction does not match the provisional API format."
    if not isinstance(payload, dict) or not isinstance(payload.get("predicted_route"), dict):
        raise BackendError(message)
    geometry = payload["predicted_route"]
    coordinates = geometry.get("coordinates")
    lines = [coordinates] if geometry.get("type") == "LineString" else coordinates
    if (
        geometry.get("type") not in {"LineString", "MultiLineString"}
        or not isinstance(lines, list) or not lines
    ):
        raise BackendError(message)
    for line in lines:
        if not isinstance(line, list) or len(line) < 2:
            raise BackendError(message)
        for point in line:
            if (
                not isinstance(point, list) or len(point) != 2
                or not all(numeric(axis) for axis in point)
                or abs(point[0]) > 180 or abs(point[1]) > 90
            ):
                raise BackendError(message)
    metrics = payload.get("metrics")
    timings = payload.get("timings_ms")
    if not isinstance(metrics, dict) or not isinstance(timings, dict):
        raise BackendError(message)
    if not numeric(metrics.get("accuracy")) or not 0 <= metrics["accuracy"] <= 1:
        raise BackendError("Backend did not return accuracy as a fraction from 0 to 1.")
    if not numeric(timings.get("inference")) or timings["inference"] < 0:
        raise BackendError("Backend did not return a valid inference time in milliseconds.")
    rows = [
        [label, metrics[key]]
        for key, label in METRICS.items()
        if numeric(metrics.get(key))
    ]
    rows.extend(
        [label, timings[key]]
        for key, label in {"inference": "Inference (ms)", "total": "Total time (ms)"}.items()
        if numeric(timings.get(key)) and timings[key] >= 0
    )
    return {"map_html": prediction_map(lines), "metrics": rows}


class BackendClient:
    def __init__(self, url):
        self.url = url
        self.opener = build_opener(ProxyHandler({}))

    def request(self, endpoint, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(self.url + endpoint, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=60 if data is not None else 5) as response:
                if endpoint == "/health":
                    return None
                return json.load(response)
        except HTTPError as error:
            error.close()
            raise BackendError(f"Backend returned HTTP {error.code} for {endpoint}.") from error
        except (URLError, OSError) as error:
            raise BackendError("Could not reach Laptop 1, or the request timed out. Check its address and connection.") from error
        except (ValueError, UnicodeError) as error:
            raise BackendError("Backend did not return valid JSON.") from error


class EvaluatorHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, backend, **kwargs):
        self.backend = backend
        super().__init__(*args, **kwargs)

    def respond(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def respond_json(self, status, payload):
        self.respond(status, "application/json", json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            self.respond(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif self.path == "/api/trips":
            try:
                self.backend.request("/health")
                trips = decode_trips(self.backend.request("/trips"))
                self.respond_json(200, {"trips": trips})
            except BackendError as error:
                self.respond_json(502, {"error": str(error)})
        else:
            self.respond_json(404, {"error": "Not found."})

    def do_POST(self):
        if self.path != "/api/predict":
            self.respond_json(404, {"error": "Not found."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1024 or self.headers.get_content_type() != "application/json":
                raise ValueError
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict) or not valid_id(payload.get("trip_id")):
                raise ValueError
        except (ValueError, UnicodeError):
            self.respond_json(400, {"error": "Select a valid eight-digit trip ID."})
            return
        try:
            prediction = self.backend.request("/predict", {"trip_id": payload["trip_id"]})
            self.respond_json(200, decode_prediction(prediction))
        except BackendError as error:
            self.respond_json(502, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description="Run the PathSnap evaluator interface on Laptop 2.")
    parser.add_argument("--backend-url", required=True, type=backend_url)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    handler = partial(EvaluatorHandler, backend=BackendClient(args.backend_url))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Evaluator UI: http://127.0.0.1:{server.server_port} → {args.backend_url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
