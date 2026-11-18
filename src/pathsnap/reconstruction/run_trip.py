"""Reconstruct a trip from its GPS track and directed road network."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from pathsnap.data.reader import load_trip
from pathsnap.features.candidates import CandidateConfig
from pathsnap.models.artifact import load_model_artifact
from .hmm import reconstruct_route


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--trip", required=True)
    parser.add_argument("--model", type=Path, help="Trusted PathSnap model artifact (.joblib)")
    args = parser.parse_args(argv)

    classifier = None
    candidate_config = CandidateConfig()
    if args.model is not None:
        try:
            classifier, candidate_config = load_model_artifact(args.model)
        except ValueError as error:
            parser.error(str(error))

    trip = load_trip(args.root, args.trip, include_route=False)
    result = reconstruct_route(trip, classifier=classifier, candidate_config=candidate_config)
    print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
