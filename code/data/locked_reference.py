"""Fresh ordinary-intersection holdout, generated only after configuration lock.

Classical refinement in this file is an offline reference labeler, never model inference.
"""

import json
from pathlib import Path

import numpy as np

from code.data.generate_dataset import random_curve, apply_hard_negative_transform
from code.geometry.nurbs import affine_normalize_pair, refine_intersection, deduplicate_intersections
from code.geometry.vectorized_basis import basis_matrix


def intersections(curve_a, curve_b, sample_count=121):
    parameters = np.linspace(0, 1, sample_count)
    point_a = basis_matrix(parameters) @ curve_a
    point_b = basis_matrix(parameters) @ curve_b
    direction_a = np.diff(point_a, axis=0)[:, None]
    direction_b = np.diff(point_b, axis=0)[None]
    difference = point_b[None, :-1] - point_a[:-1, None]
    determinant = direction_a[..., 0]*direction_b[..., 1] - direction_a[..., 1]*direction_b[..., 0]
    numerator_a = difference[..., 0]*direction_b[..., 1] - difference[..., 1]*direction_b[..., 0]
    numerator_b = difference[..., 0]*direction_a[..., 1] - difference[..., 1]*direction_a[..., 0]
    mask = np.abs(determinant) >= 1e-10
    ratio_a = np.divide(numerator_a, determinant, out=np.full_like(determinant, np.inf), where=mask)
    ratio_b = np.divide(numerator_b, determinant, out=np.full_like(determinant, np.inf), where=mask)
    mask &= (ratio_a >= -1e-9) & (ratio_a <= 1+1e-9) & (ratio_b >= -1e-9) & (ratio_b <= 1+1e-9)
    refined = []
    for row, column in np.argwhere(mask):
        initial = ((row+np.clip(ratio_a[row, column], 0, 1))/(sample_count-1),
                   (column+np.clip(ratio_b[row, column], 0, 1))/(sample_count-1))
        root = refine_intersection(curve_a, curve_b, initial)
        if root is not None:
            refined.append(root)
    return deduplicate_intersections(refined)


def create(path, count=400, seed=98711351):
    path = Path(path)
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    curves_a, curves_b, labels, counts = [], [], [], []
    attempts = 0
    while len(curves_a) < count:
        attempts += 1
        if attempts > count*300:
            raise RuntimeError("Failed to fill holdout strata")
        wanted = len(curves_a) % 4
        curve_a, curve_b = random_curve(rng), random_curve(rng)
        if wanted == 0 and rng.random() < 0.35:
            curve_b = apply_hard_negative_transform(curve_a, rng)
        curve_a, curve_b = affine_normalize_pair(curve_a, curve_b)
        roots = intersections(curve_a, curve_b)
        if len(roots) != wanted:
            continue
        denser = intersections(curve_a, curve_b, sample_count=241)
        if len(denser) != len(roots):
            continue
        if roots and np.max(np.abs(np.asarray(roots)-np.asarray(denser))) > 1e-6:
            continue
        label = np.zeros((3, 2))
        label[:len(roots)] = np.asarray(roots).reshape(-1, 2)
        curves_a.append(curve_a)
        curves_b.append(curve_b)
        labels.append(label)
        counts.append(len(roots))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, test_curve_a=curves_a, test_curve_b=curves_b, test_labels=labels, test_counts=counts)
    path.with_suffix(".json").write_text(json.dumps({"seed": seed, "count": count, "attempts": attempts,
        "created_after_configuration_lock": True, "sampling_checks": [121, 241],
        "limitation": "numerical labels for ordinary isolated crossings; no completeness proof"}, indent=2), encoding="utf-8")
    return path
