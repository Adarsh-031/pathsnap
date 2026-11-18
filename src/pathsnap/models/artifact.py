"""Load the classifier and candidate settings saved by PathSnap training."""

from pathlib import Path

from joblib import load

from pathsnap.features.candidates import CandidateConfig, FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION


def load_model_artifact(path: Path):
    """Load a trusted joblib artifact and verify the candidate feature contract."""
    artifact = load(path)
    if (artifact["feature_schema_version"] != FEATURE_SCHEMA_VERSION
            or artifact["feature_columns"] != FEATURE_COLUMNS):
        raise ValueError("Model artifact feature schema does not match this version of PathSnap")
    return artifact["estimator"], CandidateConfig(**artifact["candidate_config"])
