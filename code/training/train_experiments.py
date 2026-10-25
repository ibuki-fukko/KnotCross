"""Train the complete intersection prototype and its ablations.

The implementation uses NumPy only. It includes the losses used in the
research plan: heatmap supervision, unordered slot matching, a differentiable
NURBS geometric residual, and a duplicate prediction penalty.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.geometry.nurbs import curve_derivative, curve_point, curve_points, parameter_grid_residual
from code.models.mlp_numpy import DirectIntersectionModel, SharedResidualIntersectionModel, sigmoid


MAX_SLOTS = 3
SAMPLE_COUNT = 24
RESIDUAL_GRID = 12
MAP_GRID = RESIDUAL_GRID


def build_features(curves_a: np.ndarray, curves_b: np.ndarray) -> dict[str, np.ndarray]:
    parameters = np.linspace(0.0, 1.0, SAMPLE_COUNT, dtype=np.float64)
    sample_a = np.asarray([curve_points(curve, parameters).reshape(-1) for curve in curves_a])
    sample_b = np.asarray([curve_points(curve, parameters).reshape(-1) for curve in curves_b])
    residual = []
    for curve_a, curve_b in zip(curves_a, curves_b):
        _, dx, dy, distance, _ = parameter_grid_residual(curve_a, curve_b, RESIDUAL_GRID)
        proximity = np.exp(-distance / 0.05)
        residual.append(
            np.concatenate(
                [dx.reshape(-1) / 2.0, dy.reshape(-1) / 2.0, distance.reshape(-1) / 2.0, proximity.reshape(-1)]
            )
        )
    residual = np.asarray(residual, dtype=np.float64)
    direct = np.concatenate([sample_a, sample_b, residual], axis=1)
    return {"sample_a": sample_a, "sample_b": sample_b, "residual": residual, "direct": direct}


def heatmap_targets(labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    coordinates = np.linspace(0.0, 1.0, MAP_GRID, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(coordinates, coordinates, indexing="ij")
    targets = np.zeros((len(labels), MAP_GRID * MAP_GRID), dtype=np.float64)
    sigma = 0.06
    for index, (label, count) in enumerate(zip(labels, counts)):
        for point in label[: int(count)]:
            squared_distance = (grid_u - point[0]) ** 2 + (grid_v - point[1]) ** 2
            targets[index] = np.maximum(targets[index], np.exp(-squared_distance / (2.0 * sigma**2)).reshape(-1))
    return targets


def sigmoid_derivative(probability: np.ndarray) -> np.ndarray:
    return probability * (1.0 - probability)


def choose_assignment(predicted: np.ndarray, label: np.ndarray, count: int) -> tuple[int, ...]:
    if count == 0:
        return tuple()
    best_assignment = None
    best_cost = float("inf")
    for assignment in itertools.permutations(range(MAX_SLOTS), count):
        cost = 0.0
        for label_index, slot_index in enumerate(assignment):
            cost += float(np.sum((predicted[slot_index] - label[label_index]) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment
    if best_assignment is None:
        raise RuntimeError("No slot assignment found")
    return best_assignment


def loss_and_gradients(
    map_logits: np.ndarray,
    slot_logits: np.ndarray,
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    map_targets: np.ndarray,
    geometry_weight: float,
    map_weight: float = 0.25,
    duplicate_weight: float = 0.15,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    batch_size = len(curve_a)
    map_probability = sigmoid(map_logits)
    slot_probability = sigmoid(slot_logits.reshape(batch_size, MAX_SLOTS, 3))
    predicted_parameters = slot_probability[:, :, :2]
    predicted_existence = slot_probability[:, :, 2]

    map_difference = map_probability - map_targets
    map_loss = float(np.mean(map_difference * map_difference))
    map_gradient = map_weight * (2.0 / map_difference.size) * map_difference * sigmoid_derivative(map_probability)

    parameter_gradient = np.zeros_like(predicted_parameters)
    existence_gradient = np.zeros_like(predicted_existence)
    parameter_loss = 0.0
    existence_loss = 0.0
    geometry_loss = 0.0
    duplicate_loss = 0.0
    matched_total = 0
    for sample_index in range(batch_size):
        count = int(counts[sample_index])
        assignment = choose_assignment(predicted_parameters[sample_index], labels[sample_index], count)
        target_existence = np.zeros(MAX_SLOTS, dtype=np.float64)
        for label_index, slot_index in enumerate(assignment):
            target_existence[slot_index] = 1.0
            difference = predicted_parameters[sample_index, slot_index] - labels[sample_index, label_index]
            parameter_loss += float(np.sum(difference * difference))
            parameter_gradient[sample_index, slot_index] += 2.0 * difference
            parameter_a, parameter_b = predicted_parameters[sample_index, slot_index]
            point_a = curve_point(curve_a[sample_index], parameter_a)
            point_b = curve_point(curve_b[sample_index], parameter_b)
            geometric_difference = point_a - point_b
            geometry_loss += float(np.sum(geometric_difference * geometric_difference))
            parameter_gradient[sample_index, slot_index, 0] += geometry_weight * 2.0 * float(
                geometric_difference @ curve_derivative(curve_a[sample_index], parameter_a)
            )
            parameter_gradient[sample_index, slot_index, 1] -= geometry_weight * 2.0 * float(
                geometric_difference @ curve_derivative(curve_b[sample_index], parameter_b)
            )
            matched_total += 1
        probability = np.clip(predicted_existence[sample_index], 1e-7, 1.0 - 1e-7)
        existence_loss += float(
            -np.sum(target_existence * np.log(probability) + (1.0 - target_existence) * np.log(1.0 - probability))
        )
        existence_gradient[sample_index] = probability - target_existence

        for first_slot in range(MAX_SLOTS):
            for second_slot in range(first_slot + 1, MAX_SLOTS):
                existence_factor = predicted_existence[sample_index, first_slot] * predicted_existence[sample_index, second_slot]
                parameter_difference = predicted_parameters[sample_index, first_slot] - predicted_parameters[sample_index, second_slot]
                squared_distance = float(np.sum(parameter_difference * parameter_difference))
                margin = 0.025
                if squared_distance < margin:
                    penalty = existence_factor * (margin - squared_distance) ** 2
                    duplicate_loss += penalty
                    gradient_distance = existence_factor * (-4.0 * (margin - squared_distance) * parameter_difference)
                    parameter_gradient[sample_index, first_slot] += duplicate_weight * gradient_distance
                    parameter_gradient[sample_index, second_slot] -= duplicate_weight * gradient_distance

    normalization = max(float(batch_size), 1.0)
    parameter_loss /= normalization
    existence_loss /= normalization
    geometry_loss /= max(float(matched_total), 1.0)
    duplicate_loss /= normalization
    parameter_gradient /= normalization
    existence_gradient /= normalization
    slot_probability_gradient = np.zeros_like(slot_probability)
    slot_probability_gradient[:, :, :2] = parameter_gradient * sigmoid_derivative(predicted_parameters)
    slot_probability_gradient[:, :, 2] = existence_gradient
    slot_gradient = slot_probability_gradient.reshape(batch_size, MAX_SLOTS * 3)
    total_loss = map_weight * map_loss + parameter_loss + 0.5 * existence_loss + geometry_weight * geometry_loss + duplicate_weight * duplicate_loss
    pieces = {
        "total": total_loss,
        "map": map_loss,
        "parameter": parameter_loss,
        "existence": existence_loss,
        "geometry": geometry_loss,
        "duplicate": duplicate_loss,
    }
    return total_loss, pieces, map_gradient, slot_gradient


def predict(model, model_name: str, features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "direct":
        return model.forward(features["direct"])
    return model.forward(features["sample_a"], features["sample_b"], features["residual"])


def train_model(
    model,
    model_name: str,
    data: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    geometry_weight: float,
    output_dir: Path,
) -> dict:
    rng = np.random.default_rng(seed)
    train_features = build_features(data["train_curve_a"], data["train_curve_b"])
    validation_features = build_features(data["validation_curve_a"], data["validation_curve_b"])
    train_map = heatmap_targets(data["train_labels"], data["train_counts"])
    validation_map = heatmap_targets(data["validation_labels"], data["validation_counts"])
    history = []
    update_index = 0
    best_validation = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(data["train_curve_a"]))
        epoch_pieces = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            model.zero_grad()
            batch_features = {key: value[indices].copy() for key, value in train_features.items()}
            batch_curve_a = data["train_curve_a"][indices].copy()
            batch_curve_b = data["train_curve_b"][indices].copy()
            batch_labels = data["train_labels"][indices].copy()
            swap_mask = rng.random(len(indices)) < 0.5
            if np.any(swap_mask):
                original_a = batch_features["sample_a"][swap_mask].copy()
                batch_features["sample_a"][swap_mask] = batch_features["sample_b"][swap_mask]
                batch_features["sample_b"][swap_mask] = original_a
                channel_size = RESIDUAL_GRID * RESIDUAL_GRID
                batch_features["residual"][swap_mask, : 2 * channel_size] *= -1.0
                batch_curve_a[swap_mask], batch_curve_b[swap_mask] = (
                    batch_curve_b[swap_mask].copy(),
                    batch_curve_a[swap_mask].copy(),
                )
                batch_labels[swap_mask] = batch_labels[swap_mask][:, :, ::-1]
                batch_features["direct"][swap_mask] = np.concatenate(
                    [
                        batch_features["sample_a"][swap_mask],
                        batch_features["sample_b"][swap_mask],
                        batch_features["residual"][swap_mask],
                    ],
                    axis=1,
                )
            map_logits, slot_logits = predict(model, model_name, batch_features)
            loss, pieces, map_gradient, slot_gradient = loss_and_gradients(
                map_logits,
                slot_logits,
                batch_curve_a,
                batch_curve_b,
                batch_labels,
                data["train_counts"][indices],
                train_map[indices],
                geometry_weight,
            )
            model.backward(map_gradient, slot_gradient)
            model.step(learning_rate, update_index + 1)
            update_index += 1
            epoch_pieces.append(pieces)
        validation = evaluate_loss(
            model,
            model_name,
            validation_features,
            data["validation_curve_a"],
            data["validation_curve_b"],
            data["validation_labels"],
            data["validation_counts"],
            validation_map,
            geometry_weight,
        )
        mean_train = {key: float(np.mean([piece[key] for piece in epoch_pieces])) for key in epoch_pieces[0]}
        record = {"epoch": epoch, **{f"train_{key}": value for key, value in mean_train.items()}, **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(record)
        if validation["total"] < best_validation:
            best_validation = validation["total"]
            best_state = model.state()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{model_name} epoch={epoch:03d} train={mean_train['total']:.5f} validation={validation['total']:.5f}")
    if best_state is not None:
        model.load_state(best_state)
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_dir / "weights.npz", **model.state())
    (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model": model_name, "best_validation_loss": best_validation, "history": history}


def evaluate_loss(model, model_name, features, curve_a, curve_b, labels, counts, map_targets, geometry_weight):
    map_logits, slot_logits = predict(model, model_name, features)
    _, pieces, _, _ = loss_and_gradients(
        map_logits, slot_logits, curve_a, curve_b, labels, counts, map_targets, geometry_weight
    )
    return pieces


def create_model(model_name: str, seed: int):
    residual_dim = RESIDUAL_GRID * RESIDUAL_GRID * 4
    map_dim = MAP_GRID * MAP_GRID
    if model_name == "direct":
        return DirectIntersectionModel(SAMPLE_COUNT * 2 * 2 + residual_dim, map_dim, MAX_SLOTS, seed)
    return SharedResidualIntersectionModel(SAMPLE_COUNT * 2, residual_dim, map_dim, MAX_SLOTS, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nurbs_intersections.npz")
    parser.add_argument("--output", default="results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    data_file = np.load(args.data)
    data = {key: data_file[key] for key in data_file.files}
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("shared_full", "shared", 0.7),
        ("shared_no_geometry", "shared", 0.0),
        ("shared_no_residual", "shared", 0.7),
        ("direct", "direct", 0.7),
    ]
    summaries = []
    for model_name, model_type, geometry_weight in configs:
        model = create_model(model_type, args.seed + len(summaries) * 17)
        if model_name == "shared_no_residual":
            original_build_features = build_features
            def no_residual_features(curves_a, curves_b):
                features = original_build_features(curves_a, curves_b)
                features["residual"] = np.zeros_like(features["residual"])
                features["direct"] = np.concatenate([features["sample_a"], features["sample_b"], features["residual"]], axis=1)
                return features
            globals()["build_features"] = no_residual_features
        summaries.append(
            train_model(
                model,
                model_name,
                data,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.seed + len(summaries) * 17,
                geometry_weight,
                output_dir,
            )
        )
        if model_name == "shared_no_residual":
            globals()["build_features"] = original_build_features
    (output_dir / "training_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"saved training summary: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
