"""Trip-level splitting helpers. Rows from one trip never land in two folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DEV_PARTITIONS_FOR_FITTING = ("training",)  # student test is for comparison only


def trip_group_folds(frame: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """Yield (train_idx, val_idx) positional indices grouped by trip_id.

    Trips are shuffled with a fixed seed before GroupKFold so folds are reproducible.
    Fit any learned preprocessing (scalers, encoders, imputers) inside each fold on
    train_idx only.
    """
    trips = frame["trip_id"].astype(str).to_numpy()
    uniq = np.unique(trips)
    rng = np.random.default_rng(seed)
    perm = {t: i for i, t in enumerate(rng.permutation(uniq))}
    groups = np.array([perm[t] for t in trips])
    for tr, va in GroupKFold(n_splits=n_splits).split(frame, groups=groups):
        assert not set(trips[tr]) & set(trips[va])
        yield tr, va
