"""Unrolled learned parameter correction for 2D NURBS intersections.

The model iteratively predicts parameter increments with a shared MLP. NURBS
geometry contributes a differentiable training loss through an explicit
reverse-mode gradient; inference does not use Newton updates.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments.candidate_model import candidate_features
from code.evaluation.set_metrics import match_metrics
from code.geometry.nurbs import DEGREE, KNOTS, basis_values, curve_points
from code.geometry.vectorized_basis import basis_matrix, basis_derivative_matrix
from code.models.mlp_numpy import make_mlp, sigmoid


def basis_derivative(parameter: float, degree: int = DEGREE, knots: np.ndarray = KNOTS) -> np.ndarray:
    parameter = float(np.clip(parameter, 0.0, 1.0))
    basis_count = len(knots) - degree - 1
    lower = basis_values(parameter, degree - 1, knots)
    derivative = np.zeros(basis_count, dtype=np.float64)
    for index in range(basis_count):
        left_denominator = knots[index + degree] - knots[index]
        right_denominator = knots[index + degree + 1] - knots[index + 1]
        if left_denominator > 0.0:
            derivative[index] += degree * lower[index] / left_denominator
        if right_denominator > 0.0:
            derivative[index] -= degree * lower[index + 1] / right_denominator
    return derivative


def nurbs_points_batch(control_points: np.ndarray, parameters: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    control_points = np.asarray(control_points, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if weights is None:
        weights = np.ones(control_points.shape[1], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    basis = basis_matrix(parameters) * weights[None, :]
    denominator = np.sum(basis, axis=1, keepdims=True)
    return np.einsum("bi,bij->bj", basis, control_points) / np.maximum(denominator, 1e-14)


def nurbs_derivative_batch(control_points: np.ndarray, parameters: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    control_points = np.asarray(control_points, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if weights is None:
        weights = np.ones(control_points.shape[1], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    basis = basis_matrix(parameters) * weights[None, :]
    derivative = basis_derivative_matrix(parameters) * weights[None, :]
    denominator = np.sum(basis, axis=1, keepdims=True)
    derivative_denominator = np.sum(derivative, axis=1, keepdims=True)
    numerator = np.einsum("bi,bij->bj", basis, control_points)
    derivative_numerator = np.einsum("bi,bij->bj", derivative, control_points)
    safe_denominator = np.maximum(denominator, 1e-14)
    return (derivative_numerator * safe_denominator - numerator * derivative_denominator) / (safe_denominator * safe_denominator)


def build_split(archive, split: str, grid_size: int, max_candidates: int, local_slack: float):
    curves_a = archive[f"{split}_curve_a"]
    curves_b = archive[f"{split}_curve_b"]
    labels = archive[f"{split}_labels"]
    counts = archive[f"{split}_counts"]
    rows_x = []
    rows_y = []
    rows_a = []
    rows_b = []
    rows_group = []
    initial_coverage = 0
    total_truth = int(np.sum(counts))
    for group, (curve_a, curve_b, label, count) in enumerate(zip(curves_a, curves_b, labels, counts)):
        features, centers = candidate_features(curve_a, curve_b, grid_size, local_slack, max_candidates)
        if len(features) != max_candidates:
            raise RuntimeError(f"Expected {max_candidates} candidates, received {len(features)}")
        target = np.zeros((max_candidates, 3), dtype=np.float64)
        assigned_truth: set[int] = set()
        for candidate_index, center in enumerate(centers):
            if int(count) == 0:
                continue
            distances = np.linalg.norm(label[: int(count)] - center[None, :], axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= 0.085:
                target[candidate_index, :2] = label[nearest]
                target[candidate_index, 2] = 1.0
                assigned_truth.add(nearest)
        initial_coverage += len(assigned_truth)
        rows_x.append(features)
        rows_y.append(target)
        rows_a.append(np.repeat(curve_a[None, :, :], max_candidates, axis=0))
        rows_b.append(np.repeat(curve_b[None, :, :], max_candidates, axis=0))
        rows_group.extend([group] * max_candidates)
    return {
        "features": np.concatenate(rows_x),
        "target": np.concatenate(rows_y),
        "curve_a": np.concatenate(rows_a),
        "curve_b": np.concatenate(rows_b),
        "groups": np.asarray(rows_group, dtype=np.int64),
        "curve_pairs_a": curves_a,
        "curve_pairs_b": curves_b,
        "labels": labels,
        "counts": counts,
        "candidate_truth_coverage": initial_coverage / max(total_truth, 1),
    }


class LearnedCorrectionModel:
    def __init__(self, input_dim: int = 77, seed: int = 1, step_scale: float = 0.08, reference_coordinates: bool = False, dynamic_patches: bool = False):
        self.reference_coordinates = reference_coordinates
        self.dynamic_patches = dynamic_patches
        if reference_coordinates:
            input_dim += 2
        self.network = make_mlp(input_dim, [128, 128, 64], 3, np.random.default_rng(seed))
        self.step_scale = float(step_scale)

    def state(self):
        return self.network.state("network")

    def load_state(self, state):
        self.network.load_state(state, "network")


def dynamic_patch(current, curve_a, curve_b):
    offsets = np.arange(-2, 3)/47.0
    raw_a = current[:, 0, None]+offsets
    raw_b = current[:, 1, None]+offsets
    parameter_a, parameter_b = np.clip(raw_a, 0, 1), np.clip(raw_b, 0, 1)
    basis_a, basis_b = basis_matrix(parameter_a), basis_matrix(parameter_b)
    point_a = np.einsum("bki,bij->bkj", basis_a.reshape(-1, 5, 8), curve_a)
    point_b = np.einsum("bki,bij->bkj", basis_b.reshape(-1, 5, 8), curve_b)
    tangent_a = np.einsum("bki,bij->bkj", basis_derivative_matrix(parameter_a).reshape(-1, 5, 8), curve_a)
    tangent_b = np.einsum("bki,bij->bkj", basis_derivative_matrix(parameter_b).reshape(-1, 5, 8), curve_b)
    tangent_a *= ((raw_a > 0) & (raw_a < 1))[..., None]
    tangent_b *= ((raw_b > 0) & (raw_b < 1))[..., None]
    residual = point_a[:, :, None, :] - point_b[:, None, :, :]
    distance = np.linalg.norm(residual, axis=-1)
    distance_u = np.sum(residual*tangent_a[:, :, None], axis=-1)/np.maximum(distance, 1e-12)
    distance_v = -np.sum(residual*tangent_b[:, None], axis=-1)/np.maximum(distance, 1e-12)
    patch = np.stack([residual[..., 0], residual[..., 1], distance], axis=1).reshape(-1, 75)/2
    derivative_u = np.stack([np.broadcast_to(tangent_a[:, :, None, 0], distance.shape),
                            np.broadcast_to(tangent_a[:, :, None, 1], distance.shape), distance_u], axis=1).reshape(-1, 75)/2
    derivative_v = np.stack([np.broadcast_to(-tangent_b[:, None, :, 0], distance.shape),
                            np.broadcast_to(-tangent_b[:, None, :, 1], distance.shape), distance_v], axis=1).reshape(-1, 75)/2
    return patch, derivative_u, derivative_v


def unroll(model: LearnedCorrectionModel, features: np.ndarray, steps: int, training: bool = False, curve_a=None, curve_b=None):
    model.network.clear_cache()
    patch = features[:, 2:]
    current = features[:, :2].copy()
    records = []
    states = []
    for _ in range(steps):
        derivative_u = derivative_v = None
        if model.dynamic_patches:
            if curve_a is None or curve_b is None:
                raise ValueError("Dynamic patches require explicit curve control points")
            patch, derivative_u, derivative_v = dynamic_patch(current, curve_a, curve_b)
        network_input = np.concatenate([current, features[:, :2], patch], axis=1) if model.reference_coordinates else np.concatenate([current, patch], axis=1)
        raw = model.network.forward(network_input)[0]
        delta_activation = np.tanh(raw[:, :2])
        delta = model.step_scale * delta_activation
        updated_unclipped = current + delta
        updated = np.clip(updated_unclipped, 0.0, 1.0)
        clip_mask = ((updated_unclipped >= 0.0) & (updated_unclipped <= 1.0)).astype(np.float64)
        records.append((raw, delta_activation, clip_mask, derivative_u, derivative_v))
        current = updated
        states.append(current.copy())
    return current, records, states


def geometric_terms(current, curve_a, curve_b):
    point_a = nurbs_points_batch(curve_a, current[:, 0])
    point_b = nurbs_points_batch(curve_b, current[:, 1])
    derivative_a = nurbs_derivative_batch(curve_a, current[:, 0])
    derivative_b = nurbs_derivative_batch(curve_b, current[:, 1])
    residual = point_a - point_b
    return residual, derivative_a, derivative_b


def state_loss_and_gradient(current, target, curve_a, curve_b, geometry_weight, regression_weight):
    labels = target[:, 2]
    positive = labels > 0.5
    positive_count = max(int(np.sum(positive)), 1)
    gradient = np.zeros_like(current)
    if not np.any(positive):
        return 0.0, 0.0, gradient
    difference = current[positive] - target[positive, :2]
    regression_loss = regression_weight * float(np.mean(difference * difference))
    gradient[positive] += (regression_weight / positive_count) * difference
    if geometry_weight == 0:
        return regression_loss, 0.0, gradient
    selected_residual, derivative_a, derivative_b = geometric_terms(current[positive], curve_a[positive], curve_b[positive])
    geometry_loss = geometry_weight * float(np.mean(np.sum(selected_residual * selected_residual, axis=1)))
    gradient[positive, 0] += (2.0 * geometry_weight / positive_count) * np.sum(selected_residual * derivative_a, axis=1)
    gradient[positive, 1] -= (2.0 * geometry_weight / positive_count) * np.sum(selected_residual * derivative_b, axis=1)
    return regression_loss, geometry_loss, gradient


def supervision_weights(steps, deep_supervision):
    if steps < 1 or deep_supervision < 0:
        raise ValueError("steps must be positive and deep_supervision nonnegative")
    if steps == 1:
        return np.ones(1)
    return np.linspace(deep_supervision, 1.0, steps) if deep_supervision > 0 else np.r_[np.zeros(steps-1), 1.0]


def batch_loss_and_gradient(model, features, target, curve_a, curve_b, steps, geometry_weight, regression_weight, positive_weight, deep_supervision):
    current, records, states = unroll(model, features, steps, training=True, curve_a=curve_a, curve_b=curve_b)
    final_logits = records[-1][0][:, 2]
    probability = np.clip(sigmoid(final_logits), 1e-7, 1.0 - 1e-7)
    labels = target[:, 2]
    weights = 1.0 + (positive_weight - 1.0) * labels
    classification_loss = -np.mean(weights * (labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability)))
    grad_logits = weights * (probability - labels) / max(len(labels), 1)

    state_weights = supervision_weights(steps, deep_supervision)
    state_gradients = []
    regression_loss = 0.0
    geometry_loss = 0.0
    for state, state_weight in zip(states, state_weights):
        if state_weight == 0:
            state_gradients.append(np.zeros_like(state))
            continue
        state_regression, state_geometry, state_gradient = state_loss_and_gradient(
            state, target, curve_a, curve_b, geometry_weight, regression_weight
        )
        regression_loss += state_weight * state_regression
        geometry_loss += state_weight * state_geometry
        state_gradients.append(state_weight * state_gradient)

    model.network.zero_grad()
    grad_current = np.zeros_like(current)
    for step_index in range(steps - 1, -1, -1):
        grad_current += state_gradients[step_index]
        raw, delta_activation, clip_mask, derivative_u, derivative_v = records[step_index]
        grad_raw = np.zeros_like(raw)
        grad_raw[:, :2] = grad_current * clip_mask * model.step_scale * (1.0 - delta_activation * delta_activation)
        if step_index == steps - 1:
            grad_raw[:, 2] = grad_logits
        grad_input = model.network.backward(grad_raw)
        grad_current = grad_current * clip_mask + grad_input[:, :2]
        if model.dynamic_patches:
            patch_gradient = grad_input[:, 4:] if model.reference_coordinates else grad_input[:, 2:]
            grad_current[:, 0] += np.sum(patch_gradient * derivative_u, axis=1)
            grad_current[:, 1] += np.sum(patch_gradient * derivative_v, axis=1)
    return classification_loss + regression_loss + geometry_loss, classification_loss, regression_loss, geometry_loss


def predict_pair(model, curve_a, curve_b, grid_size, max_candidates, local_slack, probability_threshold, steps):
    if model.dynamic_patches and grid_size != 48:
        raise ValueError("Dynamic patch offsets currently require grid_size=48")
    features, _ = candidate_features(curve_a, curve_b, grid_size, local_slack, max_candidates)
    current, records, _ = unroll(model, features, steps,
        curve_a=np.repeat(curve_a[None], len(features), axis=0), curve_b=np.repeat(curve_b[None], len(features), axis=0))
    probability = sigmoid(records[-1][0][:, 2])
    order = np.argsort(-probability)
    predictions = []
    for index in order:
        if probability[index] < probability_threshold:
            continue
        point = current[index]
        if all(np.linalg.norm(point - np.asarray(previous)) > 0.025 for previous in predictions):
            predictions.append((float(point[0]), float(point[1])))
    return predictions


def evaluate(model, data, split, grid_size, max_candidates, local_slack, probability_threshold, steps):
    started = time.perf_counter()
    predictions = [
        predict_pair(model, a, b, grid_size, max_candidates, local_slack, probability_threshold, steps)
        for a, b in zip(data["curve_pairs_a"], data["curve_pairs_b"])
    ]
    elapsed = time.perf_counter() - started
    return predictions, {
        **match_metrics(predictions, data["labels"], data["counts"]),
        "prediction_count": int(sum(map(len, predictions))),
        "inference_seconds_total": elapsed,
        "inference_seconds_per_pair": elapsed / max(len(predictions), 1),
    }


def predict_cached(model, data, steps):
    current, records, _ = unroll(model, data["features"], steps, curve_a=data["curve_a"], curve_b=data["curve_b"])
    probability = sigmoid(records[-1][0][:, 2])
    model.network.clear_cache()
    return current, probability


def decode_cached(current, probability, groups, pair_count, threshold, suppression_radius=0.025):
    predictions = []
    for group in range(pair_count):
        indices = np.flatnonzero(groups == group)
        indices = indices[np.argsort(-probability[indices])]
        selected = []
        for index in indices:
            if probability[index] < threshold:
                continue
            point = current[index]
            if all(np.linalg.norm(point - previous) > suppression_radius for previous in selected):
                selected.append(point.copy())
        predictions.append([tuple(point) for point in selected])
    return predictions


def train(data: str, output: str, epochs: int = 30, batch_size: int = 256, learning_rate: float = 0.0005,
          seed: int = 20261026, grid_size: int = 48, max_candidates: int = 8, local_slack: float = 0.015,
          steps: int = 4, step_scale: float = 0.08, geometry_weight: float = 1.0,
          regression_weight: float = 4.0, positive_weight: float = 7.0, deep_supervision: float = 0.5, evaluate_test: bool = False,
          prepared_data=None, reference_coordinates: bool = False, dynamic_patches: bool = False):
    if dynamic_patches and grid_size != 48:
        raise ValueError("Dynamic patch offsets currently require grid_size=48")
    import code.experiments.candidate_model as candidate_module
    candidate_module.GRID_SIZE = grid_size
    candidate_module.MAX_CANDIDATES = max_candidates
    archive = np.load(data)
    if prepared_data is None:
        training_data = build_split(archive, "train", grid_size, max_candidates, local_slack)
        validation_data = build_split(archive, "validation", grid_size, max_candidates, local_slack)
    else:
        training_data, validation_data = prepared_data
    model = LearnedCorrectionModel(seed=seed, step_scale=step_scale, reference_coordinates=reference_coordinates, dynamic_patches=dynamic_patches)
    rng = np.random.default_rng(seed)
    history = []
    best_loss = float("inf")
    best_state = None
    optimizer_step = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(training_data["features"]))
        epoch_values = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            values = batch_loss_and_gradient(
                model,
                training_data["features"][indices],
                training_data["target"][indices],
                training_data["curve_a"][indices],
                training_data["curve_b"][indices],
                steps,
                geometry_weight,
                regression_weight,
                positive_weight,
                deep_supervision,
            )
            optimizer_step += 1
            model.network.step(learning_rate, 0.9, 0.999, 1e-8, optimizer_step)
            epoch_values.append(values)
        val_order = np.arange(len(validation_data["features"]))
        val_parts = []
        for start in range(0, len(val_order), batch_size):
            indices = val_order[start : start + batch_size]
            current, records, states = unroll(model, validation_data["features"][indices], steps,
                curve_a=validation_data["curve_a"][indices], curve_b=validation_data["curve_b"][indices])
            labels = validation_data["target"][indices, 2]
            state_weights = supervision_weights(steps, deep_supervision)
            probabilities = np.clip(sigmoid(records[-1][0][:, 2]), 1e-7, 1.0 - 1e-7)
            weights = 1.0 + (positive_weight - 1.0) * labels
            classification = -np.mean(weights * (labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)))
            regression = 0.0
            geometry = 0.0
            for state, state_weight in zip(states, state_weights):
                if state_weight == 0:
                    continue
                state_regression, state_geometry, _ = state_loss_and_gradient(
                    state, validation_data["target"][indices], validation_data["curve_a"][indices], validation_data["curve_b"][indices], geometry_weight, regression_weight
                )
                regression += state_weight * state_regression
                geometry += state_weight * state_geometry
            val_parts.append((classification, regression, geometry))
        validation_loss = float(sum(np.mean(part) for part in zip(*val_parts)))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean([x[0] for x in epoch_values])),
            "train_classification_loss": float(np.mean([x[1] for x in epoch_values])),
            "train_parameter_loss": float(np.mean([x[2] for x in epoch_values])),
            "train_geometry_loss": float(np.mean([x[3] for x in epoch_values])),
            "validation_loss": validation_loss,
        }
        history.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = model.state()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} train={record['train_loss']:.6f} val={validation_loss:.6f} geo={record['train_geometry_loss']:.6f}")

    model.load_state(best_state)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "weights.npz", **model.state())
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {
        "data": str(data), "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "grid_size": grid_size, "max_candidates": max_candidates,
        "local_slack": local_slack, "steps": steps, "step_scale": step_scale,
        "geometry_weight": geometry_weight, "regression_weight": regression_weight,
        "positive_weight": positive_weight,
        "reference_coordinates": reference_coordinates,
        "dynamic_patches": dynamic_patches,
        "deep_supervision": deep_supervision,
        "train_candidate_truth_coverage": training_data["candidate_truth_coverage"],
        "validation_candidate_truth_coverage": validation_data["candidate_truth_coverage"],
        "best_validation_loss": best_loss,
        "best_epoch": min(history, key=lambda item: item["validation_loss"])["epoch"],
        "evaluate_test": evaluate_test,
        "state_loss_weight_sum": float(np.sum(supervision_weights(steps, deep_supervision))),
        "autodiff": "manual reverse-mode BPTT through shared MLP correction steps and NURBS geometric residual",
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    probabilities = np.round(np.arange(0.05, 0.991, 0.01), 2)
    validation_current, validation_probability = predict_cached(model, validation_data, steps)
    np.savez_compressed(output / "validation_outputs.npz", parameters=validation_current,
                        probability=validation_probability, groups=validation_data["groups"])
    threshold_results = []
    validation_predictions_by_threshold = {}
    for threshold in probabilities:
        predictions = decode_cached(validation_current, validation_probability, validation_data["groups"],
                                    len(validation_data["counts"]), float(threshold))
        metrics = match_metrics(predictions, validation_data["labels"], validation_data["counts"])
        threshold_results.append((metrics["f1"], float(threshold), metrics))
    _, selected_threshold, validation_metrics = max(threshold_results, key=lambda item: (item[0], -item[1]))
    test_metrics = None
    if evaluate_test:
        _, test_metrics = evaluate(model, build_split(archive, "test", grid_size, max_candidates, local_slack), "test", grid_size, max_candidates, local_slack, selected_threshold, steps)
    report = {
        **metadata,
        "selected_probability_threshold": selected_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_threshold_results": [{"threshold": threshold, "f1": f1} for f1, threshold, _ in threshold_results],
    }
    final_regression, final_geometry, _ = state_loss_and_gradient(
        validation_current, validation_data["target"], validation_data["curve_a"], validation_data["curve_b"], 1.0, 4.0)
    validation_labels = validation_data["target"][:, 2]
    clipped_probability = np.clip(validation_probability, 1e-7, 1-1e-7)
    reference_weights = 1 + 6 * validation_labels
    reference_bce = -float(np.mean(reference_weights * (validation_labels*np.log(clipped_probability)
                                + (1-validation_labels)*np.log(1-clipped_probability))))
    report["validation_reference_final_loss"] = reference_bce + final_regression + final_geometry
    report["validation_reference_components"] = {"bce_weight7": reference_bce,
        "parameter_weight4": final_regression, "geometry_weight1": final_geometry}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "validation_f1": validation_metrics["f1"],
        "reference_final_loss": report["validation_reference_final_loss"], "threshold": selected_threshold}))
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/experiment_large.npz")
    parser.add_argument("--output", default="results/learned_correction/run_001")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--local-slack", type=float, default=0.015)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--step-scale", type=float, default=0.08)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=4.0)
    parser.add_argument("--positive-weight", type=float, default=7.0)
    parser.add_argument("--deep-supervision", type=float, default=0.5)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--reference-coordinates", action="store_true")
    parser.add_argument("--dynamic-patches", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()




