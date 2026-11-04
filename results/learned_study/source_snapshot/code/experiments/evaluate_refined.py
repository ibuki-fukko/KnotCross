"""Strict evaluation for learned candidates with optional geometric refinement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments.candidate_model import (
    CandidateModel,
    geometric_residuals,
    infer_candidates,
    match_metrics,
)
from code.geometry.nurbs import curve_point


def filter_by_geometry(predicted_sets, curves_a, curves_b, threshold):
    filtered = []
    for predictions, curve_a, curve_b in zip(predicted_sets, curves_a, curves_b):
        kept = []
        for parameter_a, parameter_b in predictions:
            residual = np.linalg.norm(
                curve_point(curve_a, parameter_a) - curve_point(curve_b, parameter_b)
            )
            if residual <= threshold:
                kept.append((parameter_a, parameter_b))
        filtered.append(kept)
    return filtered


def evaluate_variant(predictions, archive, split):
    return {
        **match_metrics(
            predictions,
            archive[f"{split}_labels"],
            archive[f"{split}_counts"],
        ),
        **geometric_residuals(
            predictions,
            archive[f"{split}_curve_a"],
            archive[f"{split}_curve_b"],
        ),
    }


def load_model(weights_path, seed):
    weights = np.load(weights_path)
    model = CandidateModel(77, seed=seed)
    model.load_state({key: weights[key] for key in weights.files})
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/experiment_large.npz")
    parser.add_argument("--weights", default="results/autoresearch/candidate_v7_candidates8/weights.npz")
    parser.add_argument("--output", default="results/autoresearch/refined_candidate_v7")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--local-slack", type=float, default=0.015)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--geometry-step", type=float, default=0.0005)
    parser.add_argument("--geometry-max", type=float, default=0.01)
    args = parser.parse_args()

    import code.experiments.candidate_model as candidate_module

    candidate_module.GRID_SIZE = args.grid_size
    candidate_module.MAX_CANDIDATES = args.max_candidates
    archive = np.load(args.data)
    model = load_model(args.weights, args.seed)
    probability_thresholds = np.arange(0.25, 0.951, args.threshold_step)
    geometry_thresholds = np.arange(0.0005, args.geometry_max + 1e-12, args.geometry_step)

    validation_started = time.perf_counter()
    best = None
    for probability_threshold in probability_thresholds:
        validation_refined = [
            infer_candidates(
                model,
                curve_a,
                curve_b,
                float(probability_threshold),
                args.local_slack,
                True,
                args.refine_steps,
            )
            for curve_a, curve_b in zip(archive["validation_curve_a"], archive["validation_curve_b"])
        ]
        for geometry_threshold in geometry_thresholds:
            validation_filtered = filter_by_geometry(
                validation_refined,
                archive["validation_curve_a"],
                archive["validation_curve_b"],
                float(geometry_threshold),
            )
            metrics = evaluate_variant(validation_filtered, archive, "validation")
            candidate = (
                metrics["f1"],
                -metrics["precision"],
                -float(geometry_threshold),
                float(probability_threshold),
                float(geometry_threshold),
                metrics,
            )
            if best is None or candidate > best:
                best = candidate
    validation_seconds = time.perf_counter() - validation_started
    _, _, _, selected_probability_threshold, selected_geometry_threshold, validation_metrics = best

    def predict(split, geometric_refine):
        started = time.perf_counter()
        predictions = [
            infer_candidates(
                model,
                curve_a,
                curve_b,
                selected_probability_threshold,
                args.local_slack,
                geometric_refine,
                args.refine_steps,
            )
            for curve_a, curve_b in zip(archive[f"{split}_curve_a"], archive[f"{split}_curve_b"])
        ]
        return predictions, time.perf_counter() - started

    raw_predictions, raw_seconds = predict("test", False)
    refined_predictions, refined_seconds = predict("test", True)
    filtered_predictions = filter_by_geometry(
        refined_predictions,
        archive["test_curve_a"],
        archive["test_curve_b"],
        selected_geometry_threshold,
    )
    report = {
        "data": args.data,
        "weights": args.weights,
        "grid_size": args.grid_size,
        "max_candidates": args.max_candidates,
        "local_slack": args.local_slack,
        "refine_steps": args.refine_steps,
        "selected_probability_threshold": selected_probability_threshold,
        "selected_geometry_threshold": selected_geometry_threshold,
        "validation_threshold_search_seconds": validation_seconds,
        "validation_metrics_filtered": validation_metrics,
        "test_metrics_raw": evaluate_variant(raw_predictions, archive, "test"),
        "test_metrics_refined": evaluate_variant(refined_predictions, archive, "test"),
        "test_metrics_filtered": evaluate_variant(filtered_predictions, archive, "test"),
        "test_inference_seconds_raw": raw_seconds,
        "test_inference_seconds_refined": refined_seconds,
        "test_inference_seconds_per_pair_refined": refined_seconds / max(len(refined_predictions), 1),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
