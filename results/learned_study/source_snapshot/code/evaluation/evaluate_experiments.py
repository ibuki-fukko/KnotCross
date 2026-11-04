"""Evaluate neural models and numerical baselines on held-out curve pairs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.geometry.nurbs import curve_point, find_intersections, parameter_grid_residual
from code.models.mlp_numpy import DirectIntersectionModel, SharedResidualIntersectionModel, sigmoid
from code.training.train_experiments import (
    MAP_GRID,
    MAX_SLOTS,
    RESIDUAL_GRID,
    SAMPLE_COUNT,
    build_features,
    create_model,
    heatmap_targets,
)


def load_state(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {key: archive[key] for key in archive.files}


def neural_predictions(model_name: str, model, features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "direct":
        map_logits, slot_logits = model.forward(features["direct"])
    else:
        map_logits, slot_logits = model.forward(features["sample_a"], features["sample_b"], features["residual"])
    slot_probability = sigmoid(slot_logits.reshape(len(slot_logits), MAX_SLOTS, 3))
    return sigmoid(map_logits), slot_probability


def extract_grid_predictions(curve_a: np.ndarray, curve_b: np.ndarray, refine: bool) -> list[tuple[float, float]]:
    if refine:
        return find_intersections(curve_a, curve_b, sample_count=161)
    parameters, _, _, distance, _ = parameter_grid_residual(curve_a, curve_b, RESIDUAL_GRID)
    candidates = []
    for row in range(1, RESIDUAL_GRID - 1):
        for column in range(1, RESIDUAL_GRID - 1):
            value = distance[row, column]
            neighborhood = distance[row - 1 : row + 2, column - 1 : column + 2]
            if value <= 0.12 and value <= float(np.min(neighborhood)):
                candidates.append((value, parameters[row], parameters[column]))
    output: list[tuple[float, float]] = []
    for _, parameter_a, parameter_b in sorted(candidates):
        if all(np.linalg.norm(np.asarray((parameter_a, parameter_b)) - np.asarray(existing)) > 0.12 for existing in output):
            output.append((parameter_a, parameter_b))
        if len(output) >= MAX_SLOTS:
            break
    return output


def assignment_cost(predicted: list[tuple[float, float]], truth: np.ndarray, count: int) -> tuple[float, list[tuple[int, int]]]:
    if not predicted or count == 0:
        return 0.0, []
    best = (float("inf"), [])
    number = min(len(predicted), count)
    for pred_indices in itertools.permutations(range(len(predicted)), number):
        for truth_indices in itertools.permutations(range(count), number):
            pairs = list(zip(pred_indices, truth_indices))
            cost = sum(float(np.sum((np.asarray(predicted[pred_index]) - truth[truth_index]) ** 2)) for pred_index, truth_index in pairs)
            if cost < best[0]:
                best = (cost, pairs)
    return best


def evaluate_predictions(
    name: str,
    predicted_sets: list[list[tuple[float, float]]],
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    tolerance: float = 0.035,
) -> dict:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    parameter_errors = []
    geometry_errors = []
    predicted_counts = []
    exact_count = 0
    no_intersection_false_positive = 0
    by_count = {}
    for index, predicted in enumerate(predicted_sets):
        truth_count = int(counts[index])
        truth = labels[index, :truth_count]
        predicted_counts.append(len(predicted))
        if len(predicted) == truth_count:
            exact_count += 1
        _, pairs = assignment_cost(predicted, truth, truth_count)
        matched_pred = set()
        matched_truth = set()
        for pred_index, truth_index in pairs:
            parameter_error = float(np.linalg.norm(np.asarray(predicted[pred_index]) - truth[truth_index]))
            parameter_errors.append(parameter_error)
            predicted_point_a = curve_point(curves_a[index], predicted[pred_index][0])
            predicted_point_b = curve_point(curves_b[index], predicted[pred_index][1])
            geometry_errors.append(float(np.linalg.norm(predicted_point_a - predicted_point_b)))
            if parameter_error <= tolerance:
                true_positive += 1
                matched_pred.add(pred_index)
                matched_truth.add(truth_index)
        false_positive += len(predicted) - len(matched_pred)
        false_negative += truth_count - len(matched_truth)
        if truth_count == 0 and predicted:
            no_intersection_false_positive += 1
        bucket = by_count.setdefault(str(truth_count), {"samples": 0, "true_positive": 0, "false_negative": 0})
        bucket["samples"] += 1
        bucket["true_positive"] += len(matched_truth)
        bucket["false_negative"] += truth_count - len(matched_truth)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    for bucket in by_count.values():
        bucket["recall"] = bucket["true_positive"] / max(bucket["true_positive"] + bucket["false_negative"], 1)
    return {
        "method": name,
        "samples": len(predicted_sets),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_count_accuracy": exact_count / max(len(predicted_sets), 1),
        "mean_parameter_error_matched": float(np.mean(parameter_errors)) if parameter_errors else None,
        "mean_geometry_residual_predicted_matched": float(np.mean(geometry_errors)) if geometry_errors else None,
        "no_intersection_false_positive_rate": no_intersection_false_positive / max(int(np.sum(counts == 0)), 1),
        "mean_predicted_count": float(np.mean(predicted_counts)),
        "by_true_count": by_count,
    }


def extract_slot_sets(slot_probability: np.ndarray, threshold: float = 0.5) -> list[list[tuple[float, float]]]:
    output = []
    for sample in slot_probability:
        active = [(float(slot[0]), float(slot[1])) for slot in sample if float(slot[2]) >= threshold]
        output.append(active)
    return output


def map_peak_sets(map_probability: np.ndarray, threshold: float = 0.35) -> list[list[tuple[float, float]]]:
    coordinates = np.linspace(0.0, 1.0, MAP_GRID)
    output = []
    for sample in map_probability:
        field = sample.reshape(MAP_GRID, MAP_GRID).copy()
        peaks = []
        for _ in range(MAX_SLOTS):
            row, column = np.unravel_index(int(np.argmax(field)), field.shape)
            value = float(field[row, column])
            if value < threshold:
                break
            peaks.append((float(coordinates[row]), float(coordinates[column])))
            field[max(0, row - 1) : min(MAP_GRID, row + 2), max(0, column - 1) : min(MAP_GRID, column + 2)] = -1.0
        output.append(peaks)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nurbs_intersections.npz")
    parser.add_argument("--models", default="results")
    parser.add_argument("--output", default="results/evaluation")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.data)
    curves_a = archive["test_curve_a"]
    curves_b = archive["test_curve_b"]
    labels = archive["test_labels"]
    counts = archive["test_counts"]
    features = build_features(curves_a, curves_b)
    validation_a = archive["validation_curve_a"]
    validation_b = archive["validation_curve_b"]
    validation_labels = archive["validation_labels"]
    validation_counts = archive["validation_counts"]
    validation_features = build_features(validation_a, validation_b)
    all_metrics = []
    refined = [extract_grid_predictions(a, b, True) for a, b in zip(curves_a, curves_b)]
    grid_only = [extract_grid_predictions(a, b, False) for a, b in zip(curves_a, curves_b)]
    all_metrics.append(evaluate_predictions("dense_sampling_newton_refinement", refined, curves_a, curves_b, labels, counts))
    all_metrics.append(evaluate_predictions("parameter_grid_only", grid_only, curves_a, curves_b, labels, counts))

    model_configs = [
        ("shared_full", "shared"),
        ("shared_no_geometry", "shared"),
        ("shared_no_residual", "shared"),
        ("direct", "direct"),
    ]
    for model_name, model_type in model_configs:
        model = create_model(model_type, 20261025 + model_configs.index((model_name, model_type)) * 17)
        model.load_state(load_state(Path(args.models) / model_name / "weights.npz"))
        eval_features = features
        if model_name == "shared_no_residual":
            eval_features = dict(features)
            eval_features["residual"] = np.zeros_like(features["residual"])
            eval_features["direct"] = np.concatenate([features["sample_a"], features["sample_b"], eval_features["residual"]], axis=1)
        validation_eval_features = validation_features
        if model_name == "shared_no_residual":
            validation_eval_features = dict(validation_features)
            validation_eval_features["residual"] = np.zeros_like(validation_features["residual"])
            validation_eval_features["direct"] = np.concatenate(
                [validation_features["sample_a"], validation_features["sample_b"], validation_eval_features["residual"]], axis=1
            )
        validation_map_probability, validation_slot_probability = neural_predictions(
            model_name, model, validation_eval_features
        )
        slot_thresholds = np.linspace(0.1, 0.9, 17)
        selected_slot_threshold = max(
            slot_thresholds,
            key=lambda threshold: evaluate_predictions(
                "validation",
                extract_slot_sets(validation_slot_probability, float(threshold)),
                validation_a,
                validation_b,
                validation_labels,
                validation_counts,
            )["f1"],
        )
        map_thresholds = np.linspace(0.2, 0.8, 13)
        selected_map_threshold = max(
            map_thresholds,
            key=lambda threshold: evaluate_predictions(
                "validation",
                map_peak_sets(validation_map_probability, float(threshold)),
                validation_a,
                validation_b,
                validation_labels,
                validation_counts,
            )["f1"],
        )
        map_probability, slot_probability = neural_predictions(model_name, model, eval_features)
        slot_sets = extract_slot_sets(slot_probability, float(selected_slot_threshold))
        slot_metrics = evaluate_predictions(model_name + "_slots", slot_sets, curves_a, curves_b, labels, counts)
        slot_metrics["selected_validation_existence_threshold"] = float(selected_slot_threshold)
        slot_metrics["mean_slot_existence_probability"] = float(np.mean(slot_probability[:, :, 2]))
        slot_metrics["heatmap_peak_metrics"] = evaluate_predictions(
            model_name + "_heatmap_peaks",
            map_peak_sets(map_probability, float(selected_map_threshold)),
            curves_a,
            curves_b,
            labels,
            counts,
        )
        slot_metrics["selected_validation_heatmap_threshold"] = float(selected_map_threshold)
        all_metrics.append(slot_metrics)
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "precision", "recall", "f1", "exact_count_accuracy", "mean_parameter_error_matched", "mean_geometry_residual_predicted_matched", "no_intersection_false_positive_rate", "mean_predicted_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in all_metrics:
            writer.writerow({field: record.get(field) for field in fields})
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
