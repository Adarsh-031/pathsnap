from collections.abc import Iterable, Sequence
from math import floor

import folium
from branca.element import Element

Coordinate = tuple[float, float]

GPS_COLOR = "#2563eb"
ROAD_COLOR = "#526071"
REFERENCE_COLOR = "#087f5b"
PREDICTION_COLOR = "#b45309"
ROAD_CELL_DEGREES = 0.002


def lat_lon(point: Coordinate) -> Coordinate:
    longitude, latitude = point
    return latitude, longitude


def road_cell(point: Coordinate) -> tuple[int, int]:
    longitude, latitude = point
    return floor(longitude / ROAD_CELL_DEGREES), floor(latitude / ROAD_CELL_DEGREES)


def bounds(points: Iterable[Coordinate]) -> tuple[Coordinate, Coordinate]:
    longitudes, latitudes = zip(*points)
    return (min(latitudes), min(longitudes)), (max(latitudes), max(longitudes))


def base_map(points: Iterable[Coordinate]) -> folium.Map:
    extent = bounds(points)
    result = folium.Map(location=extent[0], zoom_start=15, prefer_canvas=True)
    if extent[0] != extent[1]:
        result.fit_bounds(extent, padding=(24, 24), max_zoom=17)
    return result


def legend(result: folium.Map, title: str, layers: Sequence[tuple[str, str, bool]]) -> None:
    rows = "".join(
        f'<div style="margin-top:5px"><span style="display:inline-block;width:{8 if dot else 18}px;'
        f'height:{8 if dot else 3}px;background:{color};border-radius:4px;margin-right:6px;'
        f'vertical-align:middle"></span>{label}</div>'
        for label, color, dot in layers
    )
    result.get_root().html.add_child(Element(
        '<aside aria-label="Map legend" style="position:fixed;bottom:28px;left:12px;z-index:999;'
        'padding:10px 14px;background:#fff;color:#17243b;border:1px solid #cbd5e1;'
        'border-radius:8px;font:13px system-ui;box-shadow:0 2px 8px #0002">'
        f"<strong>{title}</strong>{rows}</aside>"
    ))


def development_map(
    trip_id: str,
    gps: Sequence[Coordinate],
    nodes: Sequence[Coordinate],
    arcs: Sequence[tuple[int, int]],
    reference: Sequence[int],
) -> tuple[folium.Map, int]:
    reference_lines = []
    previous_end = None
    for arc_id in reference:
        start, end = arcs[arc_id]
        if previous_end != start:
            reference_lines.append([nodes[start]])
        reference_lines[-1].append(nodes[end])
        previous_end = end

    points = [*gps, *(point for line in reference_lines for point in line)]
    trip_cells = {road_cell(point) for point in points}
    nearby_cells = {
        (column + horizontal, row + vertical)
        for column, row in trip_cells
        for horizontal in (-1, 0, 1)
        for vertical in (-1, 0, 1)
    }
    seen = set()
    roads = []
    for start, end in arcs:
        key = min(start, end), max(start, end)
        if key in seen:
            continue
        seen.add(key)
        start_point, end_point = nodes[start], nodes[end]
        if road_cell(start_point) in nearby_cells or road_cell(end_point) in nearby_cells:
            roads.append([lat_lon(start_point), lat_lon(end_point)])

    result = base_map(points)
    road_layer = folium.FeatureGroup(name="Road network").add_to(result)
    if roads:
        folium.PolyLine(roads, color=ROAD_COLOR, weight=1.4, opacity=0.5).add_to(road_layer)
    reference_layer = folium.FeatureGroup(name="Reference route").add_to(result)
    folium.PolyLine(
        [[lat_lon(point) for point in line] for line in reference_lines],
        color=REFERENCE_COLOR, weight=4, opacity=0.9,
    ).add_to(reference_layer)
    gps_layer = folium.FeatureGroup(name="Raw GPS observations").add_to(result)
    for index, point in enumerate(gps, start=1):
        folium.CircleMarker(
            lat_lon(point), radius=2.5, color=GPS_COLOR, weight=1,
            fill=True, fill_opacity=0.65, tooltip=f"GPS observation {index}",
        ).add_to(gps_layer)
    folium.LayerControl(collapsed=False).add_to(result)
    legend(result, f"Development trip {trip_id}", [
        ("Road network", ROAD_COLOR, False),
        ("Reference route", REFERENCE_COLOR, False),
        ("Raw GPS observations", GPS_COLOR, True),
    ])
    return result, len(roads)


def prediction_map(lines: Sequence[Sequence[Coordinate]]) -> str:
    result = base_map(point for line in lines for point in line)
    layer = folium.FeatureGroup(name="Predicted route").add_to(result)
    folium.PolyLine(
        [[lat_lon(point) for point in line] for line in lines],
        color=PREDICTION_COLOR, weight=5,
    ).add_to(layer)
    legend(result, "Evaluator map", [("Predicted route", PREDICTION_COLOR, False)])
    return result.get_root().render()
