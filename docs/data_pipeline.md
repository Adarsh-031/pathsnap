# Data pipeline: exploration, labels, candidates

All numbers below come from running the commands in section 1 on the supplied
`Student folder` (80 train + 10 student_test trips). The output files are in `reports/eda/`.

## 1. Commands (Windows, from the repo root)

```bat
set DATA=E:\pathsnap_data\Student folder
uv sync
uv run pytest -q
uv run python -m pathsnap.data.inventory --root "%DATA%" --out reports/eda
uv run python -m pathsnap.features.run_trip --root "%DATA%" --trip 00000001
uv run python -m pathsnap.features.coverage --root "%DATA%" --out reports/eda
```

| Output | Content |
|---|---|
| `reports/eda/inventory.csv` | per-trip file sizes, counts, NaN/range checks, time gaps, graph checks, route checks |
| `reports/eda/inventory_summary.txt` | partition counts and which trips trigger each flag |
| `reports/eda/label_summary.csv` | derived-label quality per trip |
| `reports/eda/candidate_coverage.csv` | candidate miss rate per trip |
| `reports/eda/trip_00000001/` | validation, coverage and small samples of candidates/labels |

## 2. Distances

Coordinates are projected per trip to the UTM zone of the GPS median
(`pathsnap.features.geometry.Projector`, `pyproj`, `always_xy=True` so the input order is
lon, lat). All distances and headings are computed in metres in that plane. A unit test
checks that 0.001° of latitude ≈ 111.2 m.

## 3. Findings (actual results)

**Trip 00000001** (train): 1,315 observations, 88,809 nodes, 165,790 arcs, 525 route
arcs. No invalid values. Timestamps strictly increasing; 98.1% of steps are 1 s; 25 steps
are longer than 1 s, and only 1 is longer than 5 s (max 16 s). 292 duplicate (from, to)
pairs. 135,752 arcs have a reverse twin (two-way roads are stored as two directed arcs).
The route has 525 unique arcs, no invalid IDs and no breaks (each arc's end node = next
arc's start node).

**All 90 trips:**
- 0 trips with non-finite or out-of-range coordinates, decreasing timestamps, invalid
  node references in `.arcs`, invalid arc IDs in `.route`, or isolated nodes.
- 47 trips have at least one gap longer than 5 s. Extreme gaps: trip 00000090 (153,064 s, about 42 h)
  and 00000078 (52,810 s). These look like separate drives joined into one trip, so the
  decoder should restart at large gaps.
- 11 trips have reference routes with breaks (consecutive route arcs not connected):
  00000002, 00000019 (3), 00000033, 00000041, 00000042, 00000069, 00000075 (2),
  00000076 (student_test), 00000080, 00000086, 00000089.
- Network size varies widely: 649 to 307,392 arcs; observations range from 262 to 19,313.

## 4. Reference-route alignment (derived labels)

`.route` lists arcs in order. It does **not** give one arc per GPS point
(trip 00000001: 1,315 points vs 525 route arcs). We therefore *derive* point labels:

**Monotone alignment.** Each observation `i` gets a route position `j(i)` with
`j(0) ≤ j(1) ≤ …` (the vehicle never moves backwards along its own route), choosing the
assignment that minimises the total point-to-arc distance. This is solved exactly by
dynamic programming in O(N·M). Aligning *positions* rather than arc IDs handles routes
that revisit an arc. Route arcs may get no observation (short arcs passed between 1 Hz
fixes).

Results:

| | ok | ambiguous | far (>30 m) | route positions with no obs |
|---|---:|---:|---:|---:|
| trip 00000001 | 990 | 325 | 0 | 67 of 525 |
| all 90 trips (227,577 obs) | 151,328 | 50,033 | 26,216 | 3,438 |

Trip 00000001: median label distance 4.3 m, 95th percentile 12.4 m, max 27.2 m.

`far` labels are concentrated in a few trips: 00000054 (17,491 of 19,313 obs, p95 210 m),
00000033 (4,160 of 5,844), 00000084 (1,064), 00000002 (1,015), 00000066 (413). For these
trips the reference route does not seem to cover large parts of the GPS track.

**Assumptions:** (1) the route is in travel order; (2) each GPS fix lies on exactly one
route arc; (3) the vehicle does not reverse along the route. The labels are our
construction, so observation-level accuracy is "accuracy against derived labels".

## 5. Candidate generation and coverage

For each valid observation: all arcs within `radius_m = 50` m (exact point-to-segment
distance, found via a KD-tree over points sampled every 10 m along arcs), sorted by
distance, keeping up to `max_candidates = 8`. This takes about 2 s for trip 00000001
(165,790 arcs).

| | labelled obs | label arc missing from candidates | of which `ok` labels | obs with no candidates |
|---|---:|---:|---:|---:|
| trip 00000001 | 1,315 | 2 | 0 | 0 |
| student_test | 19,160 | 81 | 0 | 0 |
| train | 208,417 | 24,835 | 0 | 18,230 |

On trip 00000001 the nearest candidate is the label arc for 79.5% of observations
(93.6% of `ok` labels). This is a distance-only baseline that the models should beat.

## 6. Questions for the organisers (before reporting observation-level accuracy)

1. How does the evaluator compute "prediction accuracy"? Per GPS observation (then which
   observation→arc mapping is official?), or route-level arc overlap/sequence match?
2. Is `.route` the complete traversal in order? Trips 00000054 and 00000033 have most GPS
   points more than 30 m from every route arc. Is the route partial there?
3. Two-way roads are two directed arcs. Is predicting the reverse twin counted as wrong?
4. Breaks in 11 reference routes: are these data errors or intended jumps?
5. Trips with gaps of hours (00000090, 00000078): should they be treated as separate segments?

Until these are answered we report accuracy on `ok` labels, plus all labels, and state
that the labels are derived.
