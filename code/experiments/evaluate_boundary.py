"""Evaluate the retained candidate model on boundary cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments.candidate_model import CandidateModel, infer_candidates, match_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/boundary_cases.npz")
    parser.add_argument("--weights", default="results/autoresearch/candidate_v3_softmin/weights.npz")
    parser.add_argument("--output", default="results/autoresearch/boundary_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    archive = np.load(args.data)
    model = CandidateModel(2 + 3 * 25 + 2 + 2 + 2 + 2, seed=20261025)
    state_archive = np.load(args.weights)
    model.load_state({key: state_archive[key] for key in state_archive.files})
    predictions = [infer_candidates(model, a, b, args.threshold) for a, b in zip(archive["curve_a"], archive["curve_b"])]
    records = {}
    metadata = json.loads(Path(args.data).with_suffix(".json").read_text(encoding="utf-8"))["metadata"]
    for case_type in ["near", "endpoint", "tangent"]:
        indices = [index for index, item in enumerate(metadata) if item["case_type"] == case_type]
        records[case_type] = match_metrics(
            [predictions[index] for index in indices],
            archive["labels"][indices],
            archive["counts"][indices],
        )
    Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
