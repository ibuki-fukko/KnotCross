"""Evaluate learned intersection candidates on exploratory boundary cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import code.experiments.candidate_model as candidate_module
from code.experiments.candidate_model import CandidateModel, geometric_residuals, infer_candidates, match_metrics
from code.geometry.nurbs import curve_point


def filter_by_geometry(predicted_sets, curves_a, curves_b, threshold):
    result = []
    for predictions, curve_a, curve_b in zip(predicted_sets, curves_a, curves_b):
        kept = []
        for parameter_a, parameter_b in predictions:
            residual = np.linalg.norm(curve_point(curve_a, parameter_a) - curve_point(curve_b, parameter_b))
            if residual <= threshold:
                kept.append((parameter_a, parameter_b))
        result.append(kept)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/boundary_cases.npz")
    parser.add_argument("--weights", default="results/autoresearch/candidate_v7_candidates8/weights.npz")
    parser.add_argument("--output", default="results/autoresearch/boundary_refined_metrics.json")
    parser.add_argument("--probability-threshold", type=float, default=0.35)
    parser.add_argument("--geometry-threshold", type=float, default=0.0005)
    parser.add_argument("--refine-steps", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--local-slack", type=float, default=0.015)
    args = parser.parse_args()
    candidate_module.GRID_SIZE = args.grid_size
    candidate_module.MAX_CANDIDATES = args.max_candidates
    archive = np.load(args.data)
    weights = np.load(args.weights)
    model = CandidateModel(77, seed=20261025)
    model.load_state({key: weights[key] for key in weights.files})
    metadata = json.loads(Path(args.data).with_suffix(".json").read_text(encoding="utf-8"))["metadata"]
    all_predictions = [
        infer_candidates(model, curve_a, curve_b, args.probability_threshold, args.local_slack, True, args.refine_steps)
        for curve_a, curve_b in zip(archive["curve_a"], archive["curve_b"])
    ]
    filtered = filter_by_geometry(all_predictions, archive["curve_a"], archive["curve_b"], args.geometry_threshold)
    records = {}
    for case_type in ["near", "endpoint", "tangent"]:
        indices = [index for index, item in enumerate(metadata) if item["case_type"] == case_type]
        records[case_type] = {
            "refined": {**match_metrics([all_predictions[index] for index in indices], archive["labels"][indices], archive["counts"][indices]), **geometric_residuals([all_predictions[index] for index in indices], archive["curve_a"][indices], archive["curve_b"][indices])},
            "filtered": {**match_metrics([filtered[index] for index in indices], archive["labels"][indices], archive["counts"][indices]), **geometric_residuals([filtered[index] for index in indices], archive["curve_a"][indices], archive["curve_b"][indices])},
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"settings": vars(args), "records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"settings": vars(args), "records": records}, indent=2))


if __name__ == "__main__":
    main()
