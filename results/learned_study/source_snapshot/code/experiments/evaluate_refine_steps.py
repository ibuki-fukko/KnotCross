"""Controlled ablation of the number of local geometric correction steps."""

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
    filtered = []
    for predictions, curve_a, curve_b in zip(predicted_sets, curves_a, curves_b):
        kept = []
        for parameter_a, parameter_b in predictions:
            residual = np.linalg.norm(curve_point(curve_a, parameter_a) - curve_point(curve_b, parameter_b))
            if residual <= threshold:
                kept.append((parameter_a, parameter_b))
        filtered.append(kept)
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/experiment_large.npz")
    parser.add_argument("--weights", default="results/autoresearch/candidate_v7_candidates8/weights.npz")
    parser.add_argument("--output", default="results/autoresearch/refine_steps_ablation.json")
    parser.add_argument("--probability-threshold", type=float, default=0.35)
    parser.add_argument("--geometry-threshold", type=float, default=0.0055)
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
    records = []
    for refine_steps in [5, 6, 8, 10]:
        predictions = [
            infer_candidates(
                model,
                curve_a,
                curve_b,
                args.probability_threshold,
                args.local_slack,
                refine_steps > 0,
                refine_steps,
            )
            for curve_a, curve_b in zip(archive["test_curve_a"], archive["test_curve_b"])
        ]
        filtered = filter_by_geometry(
            predictions,
            archive["test_curve_a"],
            archive["test_curve_b"],
            args.geometry_threshold,
        )
        records.append({
            "refine_steps": refine_steps,
            "raw": {**match_metrics(predictions, archive["test_labels"], archive["test_counts"]), **geometric_residuals(predictions, archive["test_curve_a"], archive["test_curve_b"])},
            "filtered": {**match_metrics(filtered, archive["test_labels"], archive["test_counts"]), **geometric_residuals(filtered, archive["test_curve_a"], archive["test_curve_b"])},
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"settings": vars(args), "records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"settings": vars(args), "records": records}, indent=2))


if __name__ == "__main__":
    main()

