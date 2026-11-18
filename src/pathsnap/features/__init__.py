"""Inference-safe candidate generation and feature engineering (never reads .route)."""

from .candidates import (FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, KEY_COLUMNS,
                         CandidateConfig, build_candidates)

__all__ = ["FEATURE_COLUMNS", "FEATURE_SCHEMA_VERSION", "KEY_COLUMNS",
           "CandidateConfig", "build_candidates"]
