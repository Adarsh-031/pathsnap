# PathSnap data contract (v1.0)

Owner: data & features (branch `feature/data-features`). Changes to this contract are
announced to the team and bump `FEATURE_SCHEMA_VERSION` in
`src/pathsnap/features/candidates.py`.

## 1. Raw files (verified on all 90 development trips)

Layout: `<data root>/<partition>/<trip_id>/<trip_id>.<ext>`, partitions `train` (80) and
`student_test` (10). All files are whitespace-separated with no header.

| File | Columns | ID rule |
|---|---|---|
| `.track` | longitude, latitude, timestamp_s | row index = `observation_index` |
| `.nodes` | longitude, latitude | row index = `node_id` |
| `.arcs` | from_node_id, to_node_id | row index = `arc_id` (directed) |
| `.route` | arc_id | ordered reference route — **not** one row per GPS point |

IDs are local to each trip. Coordinates are **(lon, lat)**; Folium wants `[lat, lon]`.

## 2. Trip object — `pathsnap.data.reader.Trip`

```python
from pathsnap.data import load_trip
trip = load_trip(DATA_ROOT, "00000001", include_route=False)   # inference
trip = load_trip(DATA_ROOT, "00000001")                         # dev: also loads .route
```

| Field | Type | Columns |
|---|---|---|
| `trip_id` | str | zero-padded, e.g. `"00000001"` |
| `partition` | str | `train`, `student_test` (folder name, lower-case) |
| `observations` | DataFrame | `observation_index, lon, lat, t` |
| `nodes` | DataFrame | `node_id, lon, lat` |
| `arcs` | DataFrame | `arc_id, from_node, to_node` |
| `route` | ndarray or None | reference arc IDs; `None` when `include_route=False` |

`trip.without_route()` returns a copy with no route and no route file path.

## 3. Observation table — `build_candidates(...)[0]`

One row per GPS observation, **including observations with no candidates**.

`trip_id, observation_index, lon, lat, t, x_m, y_m, dt_prev_s, speed_mps, heading_deg,
heading_valid, gap_flag, n_candidates, has_candidates`

`x_m, y_m` are metres in the trip's UTM zone (EPSG 326xx/327xx chosen from the GPS median).

## 4. Candidate records — `build_candidates(...)[1]`

One row per (observation, candidate arc). Key columns are always first:

`KEY_COLUMNS = ["trip_id", "observation_index", "arc_id"]`

Then the ordered feature schema `FEATURE_COLUMNS` (v1.0). Models must select features by
this list, in this order:

| # | Feature | Meaning |
|---|---|---|
| 1 | `dist_m` | perpendicular distance GPS point → arc segment, metres |
| 2 | `dist_minus_min_m` | `dist_m` minus nearest candidate's `dist_m` for that observation |
| 3 | `cand_rank` | 0 = nearest |
| 4 | `n_candidates` | candidates kept for that observation |
| 5 | `proj_frac` | 0–1 position of the projection along the arc (from → to) |
| 6 | `arc_length_m` | arc length, metres |
| 7 | `heading_diff_deg` | \|GPS heading − arc bearing\| in [0, 180]; `90.0` if `heading_valid == 0` |
| 8 | `heading_valid` | 1 if the vehicle moved ≥ 2 m between neighbouring fixes |
| 9 | `speed_mps` | speed from the previous fix (0 for the first) |
| 10 | `dt_prev_s` | seconds since previous fix (0 for the first) |
| 11 | `gap_flag` | 1 if `dt_prev_s > 5` |
| 12 | `from_node_in_degree` | in-degree of the arc's start node |
| 13 | `to_node_out_degree` | out-degree of the arc's end node |

Defaults: `radius_m = 50`, `max_candidates = 8` (`CandidateConfig`). The heading fill
value `90.0` is a fixed constant, not learned, so it causes no fold leakage.

**Missing candidates:** an observation with no arc within `radius_m` has
`n_candidates = 0`, `has_candidates = 0` in the observation table and **no rows** in the
candidate table. Downstream code (HMM) must handle these, e.g. by breaking the sequence.

## 5. Reference-label records (development only) — `pathsnap.data.labels`

Labels live in a **separate** table and never in the feature matrix:

`trip_id, observation_index, label_arc_id, route_position, label_dist_m,
nearest_nonroute_dist_m, near_route_boundary, label_status, label_source`

- `label_source = "derived_monotone_route_alignment_v1"`: these are **derived** by us,
  not supplied by the organisers. The supplied ground truth is only the `.route` sequence.
- `label_status`:
  - `ok`: label arc within 30 m and no competing non-route arc within +3 m
  - `ambiguous`: a non-route arc (not the label arc's reverse twin) is within 3 m of the label distance
  - `far`: the aligned route arc is more than 30 m away (the route may not cover this point)
- `near_route_boundary = 1`: the next/previous route arc is within 3 m too (junction).

Training target: `attach_training_target(candidates, labels)` returns a separate Series
`y` (1 = candidate arc equals derived label arc). Recommended: train and score on `ok`
labels; report `ambiguous`/`far` separately.

## 6. Rules

- Inference (`api`, decoder input) uses `load_trip(..., include_route=False)` and
  `build_candidates`. `build_candidates` calls `trip.without_route()` itself.
- Never import `pathsnap.data.labels` from inference/API code.
- Cross-validation: `pathsnap.data.splits.trip_group_folds` (whole trips per fold).
  Fit scalers etc. inside each fold on the training indices only.
- Evaluator-test trips are never used for fitting, tuning, thresholds or feature choices.
