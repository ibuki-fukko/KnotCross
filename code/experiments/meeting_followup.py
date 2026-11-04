"""Reproducible mechanism study for the 2026-11-04 advisor meeting.

The MLP receives raw cubic B-spline control points plus one parameter-grid
query (u0, v0), and directly predicts a final parameter pair and a score.
An ablation appends a local residual patch to the same first-layer input.
No Newton updates or geometry-threshold filtering are used at inference.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.evaluation.set_metrics import match_metrics, matching
from code.geometry.nurbs import KNOTS, affine_normalize_pair
from code.geometry.vectorized_basis import basis_derivative_matrix, basis_matrix
from code.models.mlp_numpy import make_mlp, sigmoid

OUT = ROOT / "results" / "meeting_followup"
DATA = ROOT / "data" / "experiment_large.npz"
SEED = 20261026
GEOMETRIC_TOLERANCES = [1e-3, 1e-4, 1e-5, 1e-6]
PARAMETER_TOLERANCES = [0.035, 0.02, 0.01, 0.005]
GRID_SIZE = 48
MAX_INTERSECTIONS = 3
MAX_OUTPUTS = 8
PATCH_RADIUS = 2
GLOBAL_RESIDUAL_GRID = 12


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_analytic_split(path: Path, per_type: int, seed: int) -> Path:
    """Generate exact-root cubic cases, including endpoints and near misses."""
    if path.exists() and path.with_suffix(".json").exists():
        return path
    rng = np.random.default_rng(seed)
    greville = np.array([np.mean(KNOTS[i + 1:i + 4]) for i in range(8)])
    collocation = basis_matrix(greville)
    curve_a_all, curve_b_all, labels_all, counts_all, metadata = [], [], [], [], []
    categories = ("single", "multiple", "close_0p02", "close_0p005", "endpoint", "tangent", "near_disjoint")
    for category in categories:
        for index in range(per_type):
            if category == "single":
                roots = [float(rng.uniform(0.08, 0.92))]
                polynomial_roots = roots
            elif category == "multiple":
                roots = sorted(rng.uniform(0.08, 0.92, size=3).tolist())
                while min(np.diff(roots)) < 0.12:
                    roots = sorted(rng.uniform(0.08, 0.92, size=3).tolist())
                polynomial_roots = roots
            elif category.startswith("close"):
                gap = 0.02 if category.endswith("0p02") else 0.005
                center = float(rng.uniform(0.25, 0.75))
                roots = [center - gap / 2, center + gap / 2]
                polynomial_roots = roots
            elif category == "endpoint":
                root = float(index % 2)
                roots = [root]
                polynomial_roots = roots
            elif category == "tangent":
                root = float(rng.uniform(0.1, 0.9))
                roots = [root]
                polynomial_roots = [root, root]
            else:
                roots = []
                gap = float(10.0 ** rng.uniform(-7.0, -2.0))
                polynomial_roots = None

            amplitude = float(rng.uniform(0.7, 1.5))
            if category == "near_disjoint":
                vertical = amplitude * (greville - 0.5) ** 2 + gap
            else:
                vertical = amplitude * np.prod(greville[:, None] - np.asarray(polynomial_roots)[None, :], axis=1)
            curve_a = np.linalg.solve(collocation, np.column_stack((greville, np.zeros(8))))
            curve_b = np.linalg.solve(collocation, np.column_stack((greville, vertical)))
            angle = float(rng.uniform(-np.pi, np.pi))
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            translation = rng.uniform(-0.2, 0.2, 2)
            curve_a, curve_b = affine_normalize_pair(curve_a @ rotation.T + translation,
                                                      curve_b @ rotation.T + translation)
            if roots:
                assert np.max(np.abs(basis_matrix(np.asarray(roots)) @ (curve_a - curve_b))) < 1e-10
            label = np.zeros((MAX_INTERSECTIONS, 2), dtype=np.float64)
            if roots:
                label[:len(roots)] = np.repeat(np.asarray(roots)[:, None], 2, axis=1)
            curve_a_all.append(curve_a)
            curve_b_all.append(curve_b)
            labels_all.append(label)
            counts_all.append(len(roots))
            metadata.append({"category": category, "index": index, "roots": roots,
                             "unscaled_gap": gap if category == "near_disjoint" else None})

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, curve_a=np.asarray(curve_a_all), curve_b=np.asarray(curve_b_all),
                        labels=np.asarray(labels_all), counts=np.asarray(counts_all, dtype=np.int64))
    write_json(path.with_suffix(".json"), {"seed": seed, "per_type": per_type,
        "degree": 3, "control_points": 8, "knot_vector": KNOTS.tolist(), "weights": "all ones",
        "label_source": "known real roots of a polynomial represented exactly in the cubic spline space",
        "metadata": metadata})
    return path


def load_pair_dataset(base_split: str, analytic_path: Path):
    archive = np.load(DATA)
    curves_a = [*archive[f"{base_split}_curve_a"]]
    curves_b = [*archive[f"{base_split}_curve_b"]]
    labels = [*archive[f"{base_split}_labels"]]
    counts = [*archive[f"{base_split}_counts"]]
    categories = ["interior" if int(n) else "disjoint" for n in counts]
    analytic = np.load(analytic_path)
    meta = json.loads(analytic_path.with_suffix(".json").read_text(encoding="utf-8"))["metadata"]
    curves_a.extend(analytic["curve_a"])
    curves_b.extend(analytic["curve_b"])
    labels.extend(analytic["labels"])
    counts.extend(analytic["counts"])
    categories.extend(item["category"] for item in meta)
    return {"curve_a": np.asarray(curves_a), "curve_b": np.asarray(curves_b),
            "labels": np.asarray(labels), "counts": np.asarray(counts),
            "categories": np.asarray(categories, dtype=object)}


def geometric_residual(curve_a, curve_b, parameters):
    u, v = np.asarray(parameters)[:, 0], np.asarray(parameters)[:, 1]
    ba, bb = basis_matrix(u), basis_matrix(v)
    pa = np.einsum("bi,bij->bj", ba, curve_a)
    pb = np.einsum("bi,bij->bj", bb, curve_b)
    da = np.einsum("bi,bij->bj", basis_derivative_matrix(u), curve_a)
    db = np.einsum("bi,bij->bj", basis_derivative_matrix(v), curve_b)
    return pa - pb, da, db


def local_residual_patch(curve_a, curve_b, anchors, grid_size=GRID_SIZE):
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
    offsets = np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, dtype=float) / (grid_size - 1)
    raw_u = anchors[:, 0, None] + offsets
    raw_v = anchors[:, 1, None] + offsets
    u, v = np.clip(raw_u, 0, 1), np.clip(raw_v, 0, 1)
    ba = basis_matrix(u.reshape(-1)).reshape(-1, 5, 8)
    bb = basis_matrix(v.reshape(-1)).reshape(-1, 5, 8)
    pa = np.einsum("bki,bij->bkj", ba, np.broadcast_to(curve_a, (len(anchors), 8, 2)))
    pb = np.einsum("bki,bij->bkj", bb, np.broadcast_to(curve_b, (len(anchors), 8, 2)))
    diff = pa[:, :, None, :] - pb[:, None, :, :]
    patch = np.stack([diff[..., 0], diff[..., 1], np.linalg.norm(diff, axis=-1)], axis=1)
    return (patch.reshape(len(anchors), -1) / 2.0).astype(np.float64)


def input_features(curve_a, curve_b, anchors, with_residual):
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    raw = np.concatenate([np.broadcast_to(curve_a.reshape(1, -1), (len(anchors), 16)),
                          np.broadcast_to(curve_b.reshape(1, -1), (len(anchors), 16)), anchors], axis=1)
    mode = "local" if with_residual is True else with_residual
    if mode == "local":
        raw = np.concatenate([raw, local_residual_patch(curve_a, curve_b, anchors)], axis=1)
    elif mode == "global":
        t = np.linspace(0, 1, GLOBAL_RESIDUAL_GRID)
        uu, vv = np.meshgrid(t, t, indexing="ij")
        ba, bb = basis_matrix(uu.ravel()), basis_matrix(vv.ravel())
        pa, pb = ba @ curve_a, bb @ curve_b
        diff = pa - pb
        field = np.column_stack([diff[:, 0], diff[:, 1], np.linalg.norm(diff, axis=1)]).reshape(-1) / 2
        raw = np.concatenate([raw, np.broadcast_to(field, (len(anchors), len(field)))], axis=1)
    return raw


def make_query_data(dataset, seed, queries_per_pair=16, positive_radius=0.055):
    rng = np.random.default_rng(seed)
    features, targets, curves_a, curves_b, groups = [], [], [], [], []
    for pair_index, (curve_a, curve_b, label, count) in enumerate(zip(
            dataset["curve_a"], dataset["curve_b"], dataset["labels"], dataset["counts"])):
        anchors = []
        for query_index in range(queries_per_pair):
            if int(count) and query_index < queries_per_pair // 2:
                root = label[int(rng.integers(int(count)))].copy()
                anchor = np.clip(root + rng.uniform(-positive_radius, positive_radius, 2), 0, 1)
            else:
                anchor = rng.uniform(0, 1, 2)
            anchors.append(anchor)
        anchors = np.asarray(anchors)
        target = np.zeros((queries_per_pair, 3), dtype=np.float64)
        for i, anchor in enumerate(anchors):
            if int(count):
                distances = np.linalg.norm(label[:int(count)] - anchor[None, :], axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] <= positive_radius:
                    target[i, :2] = label[nearest]
                    target[i, 2] = 1.0
        features.append(input_features(curve_a, curve_b, anchors, False))
        targets.append(target)
        curves_a.append(np.broadcast_to(curve_a, (queries_per_pair, 8, 2)))
        curves_b.append(np.broadcast_to(curve_b, (queries_per_pair, 8, 2)))
        groups.extend([pair_index] * queries_per_pair)
    return {"features": np.concatenate(features), "target": np.concatenate(targets),
            "curve_a": np.concatenate(curves_a), "curve_b": np.concatenate(curves_b),
            "groups": np.asarray(groups, dtype=np.int64), "dataset": dataset,
            "positive_radius": positive_radius}


def add_residual_features(prepared, mode="local"):
    x = prepared["features"]
    patch_dim = 75 if mode == "local" else 3 * GLOBAL_RESIDUAL_GRID * GLOBAL_RESIDUAL_GRID
    patches = np.empty((len(x), patch_dim), dtype=np.float64)
    for pair_index in np.unique(prepared["groups"]):
        indices = np.flatnonzero(prepared["groups"] == pair_index)
        curve_a, curve_b = prepared["curve_a"][indices[0]], prepared["curve_b"][indices[0]]
        if mode == "local":
            patches[indices] = local_residual_patch(curve_a, curve_b, x[indices, 32:34])
        elif mode == "global":
            t = np.linspace(0, 1, GLOBAL_RESIDUAL_GRID)
            uu, vv = np.meshgrid(t, t, indexing="ij")
            pa, pb = basis_matrix(uu.ravel()) @ curve_a, basis_matrix(vv.ravel()) @ curve_b
            diff = pa - pb
            field = np.column_stack([diff[:, 0], diff[:, 1], np.linalg.norm(diff, axis=1)]).reshape(-1) / 2
            patches[indices] = np.broadcast_to(field, (len(indices), len(field)))
        else:
            raise ValueError(mode)
    prepared = dict(prepared)
    prepared["features"] = np.concatenate([x, patches], axis=1)
    return prepared


class DirectGridMLP:
    def __init__(self, input_dim, seed):
        self.network = make_mlp(input_dim, [128, 128, 64], 3, np.random.default_rng(seed))


def batch_objective(model, batch, positive_weight=5.0, parameter_weight=4.0, geometry_weight=1.0):
    model.network.clear_cache()
    raw = model.network.forward(batch["features"])[0]
    params = sigmoid(raw[:, :2])
    logits = raw[:, 2]
    probability = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    labels = batch["target"][:, 2]
    weights = 1 + (positive_weight - 1) * labels
    classification = -np.mean(weights * (labels * np.log(probability) + (1-labels) * np.log(1-probability)))
    grad_raw = np.zeros_like(raw)
    grad_raw[:, 2] = weights * (probability - labels) / max(len(labels), 1)
    positives = labels > 0.5
    parameter_loss = geometry_loss = 0.0
    if np.any(positives):
        npos = int(np.sum(positives))
        diff = params[positives] - batch["target"][positives, :2]
        parameter_loss = parameter_weight * float(np.mean(diff * diff))
        grad_params = parameter_weight * 2 * diff / (2 * npos)
        residual, da, db = geometric_residual(batch["curve_a"][positives], batch["curve_b"][positives], params[positives])
        geometry_loss = geometry_weight * float(np.mean(np.sum(residual * residual, axis=1)))
        grad_geo = np.column_stack([2 * np.sum(residual * da, axis=1), -2 * np.sum(residual * db, axis=1)])
        grad_params += geometry_weight * grad_geo / npos
        grad_raw[positives, :2] = grad_params * params[positives] * (1 - params[positives])
    model.network.backward(grad_raw)
    return classification + parameter_loss + geometry_loss, classification, parameter_loss, geometry_loss


def direct_loss_only(model, batch):
    model.network.clear_cache()
    raw = model.network.forward(batch["features"])[0]
    params = sigmoid(raw[:, :2])
    y = batch["target"][:, 2]
    probability = np.clip(sigmoid(raw[:, 2]), 1e-7, 1-1e-7)
    weights = 1 + 4*y
    loss = -np.mean(weights*(y*np.log(probability)+(1-y)*np.log(1-probability)))
    positive = y > .5
    if np.any(positive):
        loss += 4*np.mean((params[positive]-batch["target"][positive, :2])**2)
        residual, _, _ = geometric_residual(batch["curve_a"][positive], batch["curve_b"][positive], params[positive])
        loss += np.mean(np.sum(residual*residual, axis=1))
    model.network.clear_cache()
    return float(loss)


def gradient_check(dataset):
    rng = np.random.default_rng(917)
    model = DirectGridMLP(34, 917)
    count = 6
    x = rng.normal(0, .2, (count, 34))
    target = np.zeros((count, 3))
    target[:4, :2] = rng.uniform(.2, .8, (4, 2))
    target[:4, 2] = 1
    curves_a = np.broadcast_to(dataset["curve_a"][0], (count, 8, 2)).copy()
    curves_b = np.broadcast_to(dataset["curve_b"][0], (count, 8, 2)).copy()
    batch = {"features": x, "target": target, "curve_a": curves_a, "curve_b": curves_b}
    batch_objective(model, batch)
    layer = model.network.parameter_layers()[0]
    analytic = float(layer.grad_weight[0, 0])
    original = float(layer.weight[0, 0])
    epsilon = 1e-6
    layer.weight[0, 0] = original + epsilon
    plus = direct_loss_only(model, batch)
    layer.weight[0, 0] = original - epsilon
    minus = direct_loss_only(model, batch)
    layer.weight[0, 0] = original
    numeric = (plus-minus)/(2*epsilon)
    relative = abs(analytic-numeric)/max(1e-8, abs(analytic)+abs(numeric))
    result = {"checked_parameter": "first_dense.weight[0,0]", "analytic_gradient": analytic,
              "finite_difference_gradient": numeric, "relative_error": relative,
              "passed": bool(relative < 2e-3)}
    write_json(OUT / "gradient_check.json", result)
    if not result["passed"]:
        raise RuntimeError(f"MLP gradient check failed: {result}")
    return result


def evaluate_loss(model, data, batch_size=512, positive_weight=5.0):
    values = []
    for start in range(0, len(data["features"]), batch_size):
        part = {key: val[start:start+batch_size] for key, val in data.items()
                if key in ("features", "target", "curve_a", "curve_b")}
        model.network.clear_cache()
        raw = model.network.forward(part["features"])[0]
        pred = sigmoid(raw[:, :2])
        prob = np.clip(sigmoid(raw[:, 2]), 1e-7, 1-1e-7)
        y = part["target"][:, 2]
        w = 1 + (positive_weight - 1) * y
        cls = -np.mean(w * (y*np.log(prob)+(1-y)*np.log(1-prob)))
        pos = y > 0.5
        reg = geo = 0.0
        if np.any(pos):
            reg = 4 * float(np.mean((pred[pos] - part["target"][pos, :2])**2))
            residual, _, _ = geometric_residual(part["curve_a"][pos], part["curve_b"][pos], pred[pos])
            geo = float(np.mean(np.sum(residual*residual, axis=1)))
        values.append((cls, reg, geo))
    return np.mean(values, axis=0).tolist()


def fit_variant(name, train_data, valid_data, input_residual, epochs=35, batch_size=512, lr=5e-4, positive_weight=5.0):
    output = OUT / name
    output.mkdir(parents=True, exist_ok=True)
    mode = "local" if input_residual is True else input_residual
    train = add_residual_features(train_data, mode) if mode else train_data
    valid = add_residual_features(valid_data, mode) if mode else valid_data
    model = DirectGridMLP(train["features"].shape[1], SEED)
    saved_config = output / "config.json"
    if saved_config.exists() and (output / "weights.npz").exists():
        config = json.loads(saved_config.read_text(encoding="utf-8"))
        saved_mode = config.get("residual_input_mode", "local" if config.get("residual_patch_in_first_layer") else "none")
        if (config.get("seed") == SEED and config.get("epochs") == epochs
                and config.get("input_dim") == train["features"].shape[1]
                and saved_mode == (mode or "none")
                and config.get("positive_class_weight", 5.0) == positive_weight):
            config["residual_input_mode"] = mode or "none"
            config["maximum_predictions_per_pair"] = MAX_OUTPUTS
            write_json(saved_config, config)
            model.network.load_state(dict(np.load(output / "weights.npz")), "network")
            print(f"{name}: reusing verified checkpoint {config['weights_sha256']}", flush=True)
            return model
    rng = np.random.default_rng(SEED)
    history, best_loss, best_state, step = [], float("inf"), None, 0
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(train["features"]))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            ix = order[start:start+batch_size]
            batch = {key: train[key][ix] for key in ("features", "target", "curve_a", "curve_b")}
            losses = batch_objective(model, batch, positive_weight=positive_weight)
            step += 1
            model.network.step(lr, 0.9, 0.999, 1e-8, step)
            epoch_losses.append(losses)
        parts = evaluate_loss(model, valid, batch_size, positive_weight)
        val_loss = float(sum(parts))
        record = {"epoch": epoch, "train_loss": float(np.mean([x[0] for x in epoch_losses])),
                  "validation_loss": val_loss, "validation_classification": parts[0],
                  "validation_parameter": parts[1], "validation_geometry": parts[2]}
        history.append(record)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, model.network.state("network")
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{name} epoch={epoch:02d} train={record['train_loss']:.5f} val={val_loss:.5f}", flush=True)
    model.network.load_state(best_state, "network")
    np.savez_compressed(output / "weights.npz", **model.network.state("network"))
    write_json(output / "history.json", history)
    config = {"name": name, "seed": SEED, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": lr, "input_dim": int(train["features"].shape[1]),
              "positive_class_weight": positive_weight,
              "hidden_layers": [128, 128, 64], "output": ["u_final", "v_final", "existence_logit"],
              "maximum_predictions_per_pair": MAX_OUTPUTS,
              "raw_control_points": True, "residual_input_mode": mode or "none",
              "optimizer": "Adam", "positive_query_radius": train_data["positive_radius"],
              "best_epoch": min(history, key=lambda item:item["validation_loss"])["epoch"],
              "weights_sha256": sha256(output / "weights.npz")}
    write_json(output / "config.json", config)
    return model


def grid_values(kind, n=GRID_SIZE):
    if kind == "uniform":
        return np.linspace(0, 1, n)
    if kind == "cosine_endpoints":
        x = np.linspace(0, 1, n)
        return 0.5 - 0.5*np.cos(np.pi*x)
    if kind == "knot_aligned":
        breaks = np.unique(KNOTS)
        return np.concatenate([np.linspace(a, b, 11, endpoint=False) for a, b in zip(breaks[:-1], breaks[1:])] + [np.array([1.0])])
    raise ValueError(kind)


def audit_grids(datasets):
    rows = []
    roots = np.concatenate([d["labels"][i, :int(n)] for d in datasets
                            for i, n in enumerate(d["counts"]) if int(n)], axis=0)
    for kind in ("uniform", "cosine_endpoints", "knot_aligned"):
        values = grid_values(kind)
        mesh = np.stack(np.meshgrid(values, values, indexing="ij"), axis=-1).reshape(-1, 2)
        nearest = np.min(np.linalg.norm(roots[:, None, :] - mesh[None, :, :], axis=-1), axis=1)
        rows.append({"grid": kind, "nodes_per_axis": len(values), "queries_per_pair": len(mesh),
                     "mean_nearest_root_distance": float(np.mean(nearest)),
                     "p95_nearest_root_distance": float(np.quantile(nearest, .95)),
                     "coverage_by_parameter_radius": {str(t): float(np.mean(nearest <= t)) for t in PARAMETER_TOLERANCES}})
    write_json(OUT / "grid_audit.json", rows)
    return rows


def infer_one(model, curve_a, curve_b, grid_kind, with_residual, threshold, suppression):
    values = grid_values(grid_kind)
    uu, vv = np.meshgrid(values, values, indexing="ij")
    anchors = np.column_stack([uu.ravel(), vv.ravel()])
    x = input_features(curve_a, curve_b, anchors, with_residual)
    model.network.clear_cache()
    raw = model.network.forward(x)[0]
    params = sigmoid(raw[:, :2])
    scores = sigmoid(raw[:, 2])
    order = np.argsort(-scores)
    selected = []
    for i in order:
        if scores[i] < threshold:
            break
        p = params[i]
        if all(np.linalg.norm(p-q) > suppression for q in selected):
            selected.append(p.copy())
    return selected, params, scores


def grid_outputs(model, dataset, grid_kind, with_residual):
    values = grid_values(grid_kind)
    uu, vv = np.meshgrid(values, values, indexing="ij")
    anchors = np.column_stack([uu.ravel(), vv.ravel()])
    outputs = []
    for a, b in zip(dataset["curve_a"], dataset["curve_b"]):
        x = input_features(a, b, anchors, with_residual)
        model.network.clear_cache()
        raw = model.network.forward(x)[0]
        outputs.append((sigmoid(raw[:, :2]), sigmoid(raw[:, 2])))
    return outputs


def decode_outputs(outputs, threshold, suppression):
    predictions = []
    for params, scores in outputs:
        selected = []
        buckets = {}
        cell_width = max(float(suppression), 1e-12)
        for i in np.argsort(-scores):
            if scores[i] < threshold:
                break
            point = params[i]
            key = tuple(np.floor(point / cell_width).astype(int))
            nearby = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nearby.extend(buckets.get((key[0] + dx, key[1] + dy), ()))
            if all(np.linalg.norm(point - selected[j]) > suppression for j in nearby):
                selected_index = len(selected)
                selected.append(point.copy())
                buckets.setdefault(key, []).append(selected_index)
                if len(selected) >= MAX_OUTPUTS:
                    break
        predictions.append(selected)
    return predictions


def metrics_from_predictions(predictions, dataset):
    residuals = []
    per_category = {}
    for a, b, predicted in zip(dataset["curve_a"], dataset["curve_b"], predictions):
        if predicted:
            parameters = np.asarray(predicted)
            r, _, _ = geometric_residual(np.broadcast_to(a, (len(parameters), 8, 2)),
                                          np.broadcast_to(b, (len(parameters), 8, 2)), parameters)
            residuals.extend(np.linalg.norm(r, axis=1).tolist())
    all_metrics = match_metrics(predictions, dataset["labels"], dataset["counts"], tolerance=0.035)
    all_metrics["f1_by_parameter_tolerance"] = {
        str(tol): match_metrics(predictions, dataset["labels"], dataset["counts"], tolerance=tol)["f1"]
        for tol in PARAMETER_TOLERANCES}
    residuals = np.asarray(residuals)
    all_metrics["prediction_geometry_residual"] = {
        str(t): float(np.mean(residuals <= t)) if len(residuals) else None for t in GEOMETRIC_TOLERANCES}
    all_metrics["prediction_geometry_residual_median"] = float(np.median(residuals)) if len(residuals) else None
    all_metrics["prediction_geometry_residual_p95"] = float(np.quantile(residuals, .95)) if len(residuals) else None
    for category in sorted(set(dataset["categories"])):
        indices = np.flatnonzero(dataset["categories"] == category)
        sub = match_metrics([predictions[j] for j in indices], dataset["labels"][indices], dataset["counts"][indices], tolerance=.035)
        per_category[category] = {k: sub[k] for k in ("precision", "recall", "f1", "count_accuracy", "empty_false_positive_rate")}
    all_metrics["per_category"] = per_category
    all_metrics["prediction_count_total"] = int(sum(map(len, predictions)))
    return all_metrics


def model_metrics(model, dataset, grid_kind, with_residual, threshold, suppression):
    outputs = grid_outputs(model, dataset, grid_kind, with_residual)
    predictions = decode_outputs(outputs, threshold, suppression)
    return metrics_from_predictions(predictions, dataset), predictions


def load_model(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = DirectGridMLP(config["input_dim"], SEED)
    state = dict(np.load(config_path.parent / "weights.npz"))
    model.network.load_state(state, "network")
    return model


def choose_decode(model, validation, with_residual, threshold_values=None):
    best = None
    if threshold_values is None:
        threshold_values = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
    rng = np.random.default_rng(SEED + 99)
    calibration_indices = []
    for category in sorted(set(validation["categories"])):
        candidates = np.flatnonzero(validation["categories"] == category)
        calibration_indices.extend(rng.choice(candidates, min(8, len(candidates)), replace=False).tolist())
    calibration_indices = np.asarray(sorted(calibration_indices), dtype=np.int64)
    calibration = {key: value[calibration_indices] if isinstance(value, np.ndarray) and len(value) == len(validation["counts"]) else value
                   for key, value in validation.items()}
    for grid in ("uniform", "knot_aligned", "cosine_endpoints"):
        outputs = grid_outputs(model, calibration, grid, with_residual)
        for suppression in (0.005, 0.01, 0.025):
            for threshold in threshold_values:
                predictions = decode_outputs(outputs, float(threshold), suppression)
                metrics = match_metrics(predictions, calibration["labels"], calibration["counts"], tolerance=.035)
                item = {"grid": grid, "suppression_radius": suppression, "threshold": float(threshold),
                        "calibration_indices": calibration_indices.tolist(), "calibration_pairs": len(calibration_indices),
                        "validation_calibration_f1": metrics["f1"]}
                if best is None or (metrics["f1"], - (metrics["mean_parameter_error"] or 0)) > (
                        best["validation_calibration_f1"], - (best.get("validation_calibration_parameter_error") or 0)):
                    item["validation_calibration_parameter_error"] = metrics["mean_parameter_error"]
                    best = item
    outputs = grid_outputs(model, validation, best["grid"], with_residual)
    predictions = decode_outputs(outputs, best["threshold"], best["suppression_radius"])
    best["validation_full"] = metrics_from_predictions(predictions, validation)
    return best


def quality_audit(datasets):
    rows = []
    for split, dataset in datasets.items():
        residuals, endpoint_flags, counts_by_cat, near_miss_distances = [], [], {}, []
        for a, b, labels, count, category in zip(dataset["curve_a"], dataset["curve_b"], dataset["labels"], dataset["counts"], dataset["categories"]):
            counts_by_cat[str(category)] = counts_by_cat.get(str(category), 0) + 1
            for u, v in labels[:int(count)]:
                r, _, _ = geometric_residual(a[None], b[None], np.array([[u, v]]))
                residuals.append(float(np.linalg.norm(r)))
                endpoint_flags.append(min(u, 1-u, v, 1-v) <= 1e-6)
            if str(category) == "near_disjoint":
                r, _, _ = geometric_residual(a[None], b[None], np.array([[0.5, 0.5]]))
                near_miss_distances.append(float(np.linalg.norm(r)))
        rows.append({"split": split, "pairs": len(dataset["counts"]), "roots": len(residuals),
                     "category_counts": counts_by_cat,
                     "endpoint_roots_parameter_tol_1e-6": int(sum(endpoint_flags)),
                     "label_residual_quantiles": np.quantile(residuals, [0, .5, .95, .99, 1]).tolist() if residuals else [],
                     "label_quality_pass_rate": {str(t): float(np.mean(np.asarray(residuals) <= t)) for t in GEOMETRIC_TOLERANCES},
                     "near_disjoint_min_distance_quantiles": np.quantile(near_miss_distances, [0, .25, .5, .75, 1]).tolist() if near_miss_distances else [],
                     "near_disjoint_ambiguous_pairs_by_geometric_tolerance": {
                         str(t): int(np.sum(np.asarray(near_miss_distances) <= t)) for t in GEOMETRIC_TOLERANCES}})
    write_json(OUT / "data_quality.json", rows)
    return rows


def evaluate_legacy_model(dataset):
    import code.experiments.candidate_model as candidate_module
    from code.experiments.learned_correction import LearnedCorrectionModel, predict_pair

    run = ROOT / "results/learned_study/refresh_scale_0p04"
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    candidate_module.GRID_SIZE = int(config["grid_size"])
    candidate_module.MAX_CANDIDATES = int(config["max_candidates"])
    model = LearnedCorrectionModel(seed=config["seed"], step_scale=config["step_scale"],
        reference_coordinates=config.get("reference_coordinates", False),
        dynamic_patches=config.get("dynamic_patches", False))
    model.load_state(dict(np.load(run / "weights.npz")))
    predictions = [predict_pair(model, a, b, config["grid_size"], config["max_candidates"],
        config["local_slack"], 0.95, config["steps"])
        for a, b in zip(dataset["curve_a"], dataset["curve_b"])]
    metrics = metrics_from_predictions(predictions, dataset)
    return {"source_run": str(run.relative_to(ROOT)), "source_weights_sha256": sha256(run / "weights.npz"),
            "threshold": 0.95, "test": metrics, "predictions": predictions}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    analytic_train = make_analytic_split(ROOT / "data/meeting_analytic_train.npz", 120, 20261104)
    analytic_validation = make_analytic_split(ROOT / "data/meeting_analytic_validation.npz", 30, 20261105)
    analytic_test = make_analytic_split(ROOT / "data/meeting_analytic_test.npz", 30, 20261106)
    datasets = {
        "train": load_pair_dataset("train", analytic_train),
        "validation": load_pair_dataset("validation", analytic_validation),
        "test": load_pair_dataset("test", analytic_test),
    }
    quality = quality_audit(datasets)
    gradients = gradient_check(datasets["train"])
    grids = audit_grids([datasets["validation"], datasets["test"]])
    train_queries = make_query_data(datasets["train"], SEED + 1)
    valid_queries = make_query_data(datasets["validation"], SEED + 2)
    protocol = {"seed": SEED, "data": str(DATA.relative_to(ROOT)), "data_sha256": sha256(DATA),
        "split_sizes": {k: len(v["counts"]) for k, v in datasets.items()},
        "analytic_split_files": [str(p.relative_to(ROOT)) for p in (analytic_train, analytic_validation, analytic_test)],
        "grid_audit": grids, "geometric_tolerances": GEOMETRIC_TOLERANCES,
        "gradient_check": gradients,
        "parameter_metric_tolerances": PARAMETER_TOLERANCES,
        "positive_query_radius": train_queries["positive_radius"],
        "inference": "direct MLP outputs (u,v), score; no Newton and no geometric-residual filtering",
        "maximum_predictions_per_pair": MAX_OUTPUTS,
        "global_residual_grid": GLOBAL_RESIDUAL_GRID,
        "validation_only_selection": True,
        "test_access": "exploration test was read before training for grid coverage diagnostics; model selection used validation only; all test results are exploratory",
        "existing_dataset_note": "procedurally generated random cubic B-spline pairs; labels were previously polyline crossings refined offline",
        "new_data_note": "programmatically generated known-root cubic spline cases; not output from a learned generative model"}
    write_json(OUT / "protocol.json", protocol)

    models = {}
    variants = (("direct_raw", False), ("direct_raw_plus_grid_input", True),
                ("direct_raw_plus_global_grid_input", "global"))
    for name, use_residual in variants:
        model = fit_variant(name, train_queries, valid_queries, use_residual)
        models[name] = model
        selection = choose_decode(model, datasets["validation"], use_residual)
        write_json(OUT / name / "validation_decode_selection.json", selection)
        print(name, "selected", json.dumps({k: selection[k] for k in ("grid", "suppression_radius", "threshold")}), flush=True)

    results = {}
    for name, use_residual in variants:
        model = models[name]
        selection = json.loads((OUT/name/"validation_decode_selection.json").read_text(encoding="utf-8"))
        metrics, predictions = model_metrics(model, datasets["test"], with_residual=use_residual,
            grid_kind=selection["grid"], threshold=selection["threshold"], suppression=selection["suppression_radius"])
        results[name] = {"selection": selection, "test": metrics}
        write_json(OUT/name/"test_metrics.json", metrics)
        write_json(OUT/name/"test_predictions.json", [[[float(x), float(y)] for x, y in row] for row in predictions])
    legacy = evaluate_legacy_model(datasets["test"])
    write_json(OUT / "legacy_learned_correction" / "test_metrics.json",
               {k:v for k,v in legacy.items() if k != "predictions"})
    write_json(OUT / "legacy_learned_correction" / "test_predictions.json",
               [[[float(x), float(y)] for x,y in row] for row in legacy["predictions"]])
    results["legacy_learned_correction"] = {k:v for k,v in legacy.items() if k != "predictions"}
    result = {"quality_audit": quality, "grid_audit": grids, "models": results}
    write_json(OUT / "experiment_results.json", result)
    print(json.dumps({k: {"val_f1":v.get("selection", {}).get("validation_full", {}).get("f1"), "test_f1":v["test"]["f1"],
        "strict_0p01":v["test"]["f1_by_parameter_tolerance"]["0.01"],
        "geometry_pass":v["test"]["prediction_geometry_residual"]} for k,v in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
