"""Candidate-proposal experiment for learned NURBS intersection refinement."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.models.mlp_numpy import make_mlp, sigmoid
from code.geometry.nurbs import curve_points

MAX_CANDIDATES = 12
PATCH_RADIUS = 2
GRID_SIZE = 32


def residual_field(curve_a: np.ndarray, curve_b: np.ndarray, grid_size: int = GRID_SIZE):
    parameters = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    points_a = curve_points(curve_a, parameters)
    points_b = curve_points(curve_b, parameters)
    difference = points_a[:, None, :] - points_b[None, :, :]
    dx = difference[:, :, 0]
    dy = difference[:, :, 1]
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    return parameters, dx, dy, distance


def candidate_indices(distance: np.ndarray, max_candidates: int = MAX_CANDIDATES):
    rows, columns = distance.shape
    candidates = []
    for row in range(1, rows - 1):
        for column in range(1, columns - 1):
            neighborhood = distance[row - 1 : row + 2, column - 1 : column + 2]
            if distance[row, column] <= float(np.min(neighborhood)):
                candidates.append((float(distance[row, column]), row, column))
    candidates.sort()
    selected = []
    minimum_grid_distance = max(1, int(round(0.04 * rows)))
    for value, row, column in candidates:
        if all(abs(row - other_row) > minimum_grid_distance or abs(column - other_column) > minimum_grid_distance for _, other_row, other_column in selected):
            selected.append((value, row, column))
        if len(selected) >= max_candidates:
            break
    while len(selected) < max_candidates:
        selected.append((float(np.max(distance)), rows // 2, columns // 2))
    return selected


def candidate_features(curve_a: np.ndarray, curve_b: np.ndarray, grid_size: int = GRID_SIZE):
    parameters, dx, dy, distance = residual_field(curve_a, curve_b, grid_size)
    selected = candidate_indices(distance)
    features = []
    centers = []
    for _, row, column in selected:
        row_start = max(0, row - PATCH_RADIUS)
        row_end = min(grid_size, row + PATCH_RADIUS + 1)
        column_start = max(0, column - PATCH_RADIUS)
        column_end = min(grid_size, column + PATCH_RADIUS + 1)
        patch = np.zeros((3, PATCH_RADIUS * 2 + 1, PATCH_RADIUS * 2 + 1), dtype=np.float64)
        patch[0, : row_end - row_start, : column_end - column_start] = dx[row_start:row_end, column_start:column_end] / 2.0
        patch[1, : row_end - row_start, : column_end - column_start] = dy[row_start:row_end, column_start:column_end] / 2.0
        patch[2, : row_end - row_start, : column_end - column_start] = distance[row_start:row_end, column_start:column_end] / 2.0
        features.append(np.concatenate([[parameters[row], parameters[column]], patch.reshape(-1)]))
        centers.append((parameters[row], parameters[column]))
    return np.asarray(features, dtype=np.float64), np.asarray(centers, dtype=np.float64)


def build_candidate_dataset(curves_a, curves_b, labels, counts):
    all_features = []
    all_targets = []
    all_groups = []
    tolerance = 0.085
    for pair_index, (curve_a, curve_b, label, count) in enumerate(zip(curves_a, curves_b, labels, counts)):
        features, centers = candidate_features(curve_a, curve_b)
        targets = np.zeros((len(features), 3), dtype=np.float64)
        for candidate_index, center in enumerate(centers):
            if int(count) == 0:
                continue
            distances = np.linalg.norm(label[: int(count)] - center[None, :], axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= tolerance:
                targets[candidate_index, 0] = label[nearest, 0] - center[0]
                targets[candidate_index, 1] = label[nearest, 1] - center[1]
                targets[candidate_index, 2] = 1.0
        all_features.append(features)
        all_targets.append(targets)
        all_groups.extend([pair_index] * len(features))
    return np.concatenate(all_features), np.concatenate(all_targets), np.asarray(all_groups, dtype=np.int64)


class CandidateModel:
    def __init__(self, input_dim: int, seed: int = 1):
        self.network = make_mlp(input_dim, [128, 128, 64], 3, np.random.default_rng(seed))

    def forward(self, features):
        return self.network.forward(features)[0]

    def backward(self, gradient):
        self.network.backward(gradient)

    def zero_grad(self):
        self.network.zero_grad()

    def step(self, learning_rate, step_index):
        self.network.step(learning_rate, 0.9, 0.999, 1e-8, step_index)

    def state(self):
        return self.network.state("network")

    def load_state(self, state):
        self.network.load_state(state, "network")


def candidate_loss(logits, targets):
    probability = sigmoid(logits[:, 2])
    target_probability = targets[:, 2]
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    classification = -np.mean(target_probability * np.log(probability) + (1.0 - target_probability) * np.log(1.0 - probability))
    positive = target_probability > 0.5
    regression = 0.0
    gradient = np.zeros_like(logits)
    gradient[:, 2] = (probability - target_probability) / max(len(targets), 1)
    if np.any(positive):
        difference = logits[positive, :2] - targets[positive, :2]
        regression = float(np.mean(difference * difference))
        gradient[positive, :2] = 2.0 * difference / max(int(np.sum(positive)), 1)
    return classification + regression, classification, regression, gradient


def infer_candidates(model, curve_a, curve_b, threshold=0.5):
    features, centers = candidate_features(curve_a, curve_b)
    logits = model.forward(features)
    probability = sigmoid(logits[:, 2])
    predictions = []
    for index in np.argsort(-probability):
        if probability[index] < threshold:
            continue
        point = np.clip(centers[index] + logits[index, :2], 0.0, 1.0)
        if all(np.linalg.norm(point - np.asarray(previous)) > 0.025 for previous in predictions):
            predictions.append((float(point[0]), float(point[1])))
    return predictions


def match_metrics(predicted_sets, labels, counts, tolerance=0.035):
    true_positive = false_positive = false_negative = 0
    parameter_errors = []
    for predicted, label, count in zip(predicted_sets, labels, counts):
        truth = label[: int(count)]
        best_pairs = []
        best_cost = float("inf")
        for number in range(min(len(predicted), len(truth)), -1, -1):
            for predicted_indices in itertools.permutations(range(len(predicted)), number):
                for truth_indices in itertools.permutations(range(len(truth)), number):
                    pairs = list(zip(predicted_indices, truth_indices))
                    cost = sum(float(np.sum((np.asarray(predicted[p]) - truth[t]) ** 2)) for p, t in pairs)
                    if cost < best_cost:
                        best_cost = cost
                        best_pairs = pairs
                if best_pairs or number == 0:
                    break
            if best_pairs or number == 0:
                break
        matched_pred = set()
        matched_truth = set()
        for pred_index, truth_index in best_pairs:
            error = float(np.linalg.norm(np.asarray(predicted[pred_index]) - truth[truth_index]))
            if error <= tolerance:
                true_positive += 1
                parameter_errors.append(error)
                matched_pred.add(pred_index)
                matched_truth.add(truth_index)
        false_positive += len(predicted) - len(matched_pred)
        false_negative += len(truth) - len(matched_truth)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_parameter_error": float(np.mean(parameter_errors)) if parameter_errors else None,
        "predicted_count": float(np.mean([len(item) for item in predicted_sets])),
    }


def train_candidate_model(data_path: str, output_dir: str, epochs: int = 30, batch_size: int = 256, learning_rate: float = 0.001, seed: int = 20261025):
    archive = np.load(data_path)
    train_features, train_targets, _ = build_candidate_dataset(archive["train_curve_a"], archive["train_curve_b"], archive["train_labels"], archive["train_counts"])
    validation_features, validation_targets, _ = build_candidate_dataset(archive["validation_curve_a"], archive["validation_curve_b"], archive["validation_labels"], archive["validation_counts"])
    model = CandidateModel(train_features.shape[1], seed)
    rng = np.random.default_rng(seed)
    step_index = 0
    best_state = None
    best_validation = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_features))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            model.zero_grad()
            logits = model.forward(train_features[indices])
            total, classification, regression, gradient = candidate_loss(logits, train_targets[indices])
            model.backward(gradient)
            step_index += 1
            model.step(learning_rate, step_index)
            losses.append((total, classification, regression))
        validation_logits = model.forward(validation_features)
        validation_loss = candidate_loss(validation_logits, validation_targets)[0]
        record = {"epoch": epoch, "train_loss": float(np.mean([item[0] for item in losses])), "validation_loss": float(validation_loss)}
        history.append(record)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = model.state()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"candidate epoch={epoch:03d} train={record['train_loss']:.5f} validation={record['validation_loss']:.5f}")
    model.load_state(best_state)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "weights.npz", **model.state())
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    validation_threshold_results = []
    for threshold in np.linspace(0.05, 0.95, 19):
        predictions = [infer_candidates(model, a, b, float(threshold)) for a, b in zip(archive["validation_curve_a"], archive["validation_curve_b"])]
        validation_threshold_results.append((match_metrics(predictions, archive["validation_labels"], archive["validation_counts"])["f1"], float(threshold)))
    selected_threshold = max(validation_threshold_results)[1]
    validation_predictions = [infer_candidates(model, a, b, selected_threshold) for a, b in zip(archive["validation_curve_a"], archive["validation_curve_b"])]
    test_predictions = [infer_candidates(model, a, b, selected_threshold) for a, b in zip(archive["test_curve_a"], archive["test_curve_b"])]
    report = {
        "validation_loss": best_validation,
        "selected_threshold": selected_threshold,
        "validation_metrics": match_metrics(validation_predictions, archive["validation_labels"], archive["validation_counts"]),
        "test_metrics": match_metrics(test_predictions, archive["test_labels"], archive["test_counts"]),
        "train_candidates": len(train_features),
        "validation_candidates": len(validation_features),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/experiment_large.npz")
    parser.add_argument("--output", default="results/autoresearch/candidate_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261025)
    arguments = parser.parse_args()
    train_candidate_model(arguments.data, arguments.output, arguments.epochs, arguments.batch_size, arguments.learning_rate, arguments.seed)
