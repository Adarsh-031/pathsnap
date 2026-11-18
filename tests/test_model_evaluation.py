import numpy as np
import pandas as pd

from pathsnap.models.evaluation import evaluate


def test_top1_counts_missing_candidates_and_uses_nearest_for_ties():
    observations = pd.DataFrame({
        "trip_id": ["t"] * 3,
        "observation_index": [0, 1, 2],
        "label_arc_id": [10, 11, 12],
        "label_status": ["ok", "ok", "far"],
        "gap_flag": [0, 0, 0],
    })
    rows = pd.DataFrame({
        "trip_id": ["t"] * 3,
        "observation_index": [0, 0, 1],
        "arc_id": [99, 10, 98],
        "cand_rank": [1, 0, 0],
        "from_node": [2, 0, 1],
        "to_node": [3, 1, 2],
        "label_status": ["ok"] * 3,
        "y": [0, 1, 0],
    })

    metrics = evaluate(rows, observations, np.array([0.8, 0.8, 0.1]))

    assert metrics["ok"]["top1_accuracy"] == 0.5
    assert metrics["ok"]["candidate_miss_rate"] == 0.5
    assert metrics["all"]["top1_accuracy"] == 1 / 3
    assert metrics["all"]["candidate_miss_rate"] == 2 / 3
    assert metrics["far"]["no_candidate_count"] == 1
    assert metrics["ambiguous"]["top1_accuracy"] is None
    assert metrics["ok"]["classification"]["confusion_matrix"] == [[1, 1], [0, 1]]
    assert metrics["direct_continuity"] == {"scored_pairs": 1, "rate": 1.0}
