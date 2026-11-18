"""Physical-distance geometry.

Longitude/latitude are angles, so we project every trip into the UTM zone that
contains its GPS centroid (metres, conformal, distortion < 0.1% within a zone)
and do all distance/heading work in that planar frame.
"""

from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


class Projector:
    """lon/lat (degrees) -> x/y (metres) in a fixed UTM zone."""

    def __init__(self, lon0: float, lat0: float):
        self.crs = utm_crs_for(lon0, lat0)
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    @classmethod
    def for_lonlat(cls, lon: np.ndarray, lat: np.ndarray) -> "Projector":
        return cls(float(np.nanmedian(lon)), float(np.nanmedian(lat)))

    def to_xy(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        # always_xy=True -> argument order is (lon, lat)
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB, broadcasting.

    Returns (distance_m, t) where t in [0, 1] is the clamped projection fraction along AB.
    """
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / len2
    t = np.where(len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(dx, dy):
    """Compass-style bearing in planar coords: 0 = +y (north), 90 = +x (east)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)
