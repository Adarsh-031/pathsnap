# PathSnap

## Reconstruct a route

Build GPS candidates and decode a directed route with the HMM. The command writes
JSON with one match per observation and an ordered list of continuous sections.

```bash
uv run python -m pathsnap.reconstruction.run_trip --root "/path/to/Student folder" --trip 00000001
```

Pass `--model /path/to/model.joblib` to use a trusted PathSnap training artifact.
The artifact supplies the classifier and candidate-generation settings. The
default uses GPS-to-road distance for emissions.

In Python, call `pathsnap.reconstruction.reconstruct_route(trip, classifier=estimator)`
with a fitted binary classifier. For existing candidate scores, pass the observation
table, candidate table, and positive-class probabilities in candidate row order
to `pathsnap.reconstruction.decode_route` along with the trip's `RoadGraph`.

Each match gives the observation index, selected directed arc ID, and section index.
An observation without candidates has null arc and section IDs. Each section lists
the traversed arc IDs, including connecting arcs between observations. The inference
command uses the GPS track and road network.

## Decoder

Each hidden state is a candidate directed arc and the observation's projected
position on it. Emission scores are the log of the classifier probability, or
a Gaussian log score based on perpendicular distance with a 10 m standard
deviation. Viterbi adds the emission score to the best predecessor's cumulative
score plus the transition score, then traces back the best path in each section.

Transitions use the shortest directed distance between the projected positions,
including partial arcs. The log transition score is the negative absolute
difference between that distance and the observed GPS displacement, scaled by
20 m. The decoder discards routes that exceed the GPS displacement by more than
100 m. It allows up to 10 m of backward GPS jitter on the same arc. A larger
backward step requires a directed path back onto that arc.

An observation without candidates, a flagged GPS gap (over 5 seconds by default),
or a step with no reachable candidate from the active Viterbi states ends the
current section. Decoding restarts at the next observation with candidates.

## Development comparison

The comparison used the existing logistic regression artifact (selected by mean
training-fold top-1 accuracy), identical candidate probabilities for both methods,
and default decoder settings on all ten `student_test` trips.

- Pointwise accuracy on 12,281 `ok` derived labels was 94.72% for classifier
  top-1 and 93.85% after Viterbi decoding.
- Directed continuity was 92.11% for classifier top-1 and 99.83% after
  decoding, measured over 19,106 adjacent observations with candidates and no
  flagged gap. Both methods used the same directed transition constraints.
- Mean ordered arc-sequence F1 rose from 63.19% to 79.89%. For each trip,
  F1 was twice the longest common subsequence length divided by the combined
  predicted and reference route lengths. The classifier route collapses consecutive
  repeats within each uninterrupted run. The HMM route includes inferred connecting
  arcs and preserves repeat traversals.

The largest pointwise loss was on trip `00000077`, where Viterbi matched 80
fewer of the 2,222 scorable observations while improving ordered arc-sequence F1
from 26.20% to 42.17%.
