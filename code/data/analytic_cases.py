"""Exactly labelled polynomial cases in the supported cubic B-spline family."""

import json
from pathlib import Path

import numpy as np

from code.geometry.nurbs import KNOTS, affine_normalize_pair
from code.geometry.vectorized_basis import basis_matrix


def create(path, per_type=30, seed=550127):
    path = Path(path)
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    greville = np.array([np.mean(KNOTS[index+1:index+4]) for index in range(8)])
    collocation = basis_matrix(greville)
    curves_a, curves_b, labels, counts, metadata = [], [], [], [], []
    for category in ("single", "multiple", "close_0p02", "close_0p005", "endpoint", "tangent", "near_disjoint"):
        for index in range(per_type):
            if category == "single":
                roots = [rng.uniform(0.15, 0.85)]
            elif category == "multiple":
                roots = [rng.uniform(0.1, 0.25), rng.uniform(0.4, 0.6), rng.uniform(0.75, 0.9)]
            elif category.startswith("close"):
                center = rng.uniform(0.3, 0.7)
                gap = 0.02 if category == "close_0p02" else 0.005
                roots = [center-gap/2, center+gap/2]
            elif category == "endpoint":
                roots = [float(index % 2)]
            elif category == "tangent":
                roots = [rng.uniform(0.2, 0.8)]
            else:
                roots = []
            amplitude = rng.uniform(0.8, 1.5)
            if category == "near_disjoint":
                vertical = amplitude*(greville-0.5)**2+rng.uniform(0.002, 0.02)
            elif category == "tangent":
                vertical = amplitude*(greville-roots[0])**2
            else:
                vertical = amplitude*np.prod(greville[:, None]-np.asarray(roots)[None], axis=1)
            curve_a = np.linalg.solve(collocation, np.c_[greville, np.zeros(8)])
            curve_b = np.linalg.solve(collocation, np.c_[greville, vertical])
            angle = rng.uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            translation = rng.uniform(-0.2, 0.2, 2)
            curve_a, curve_b = affine_normalize_pair(curve_a@rotation.T+translation,
                                                      curve_b@rotation.T+translation)
            if roots:
                pair_error = basis_matrix(roots)@(curve_a-curve_b)
                assert np.max(np.abs(pair_error)) < 1e-11
            if category == "tangent":
                from code.geometry.vectorized_basis import basis_derivative_matrix
                assert np.max(np.abs(basis_derivative_matrix(roots)@(curve_a-curve_b))) < 1e-10
            label = np.zeros((3, 2))
            label[:len(roots)] = np.repeat(np.asarray(roots)[:, None], 2, axis=1)
            curves_a.append(curve_a)
            curves_b.append(curve_b)
            labels.append(label)
            counts.append(len(roots))
            metadata.append({"category": category, "index": index, "roots": roots})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, curve_a=curves_a, curve_b=curves_b, labels=labels, counts=counts)
    path.with_suffix(".json").write_text(json.dumps({"seed": seed, "per_type": per_type,
        "label_source": "exact roots of polynomial y(v); x(u)=u and x(v)=v",
        "offline_linear_solve": "only converts polynomial coefficients into spline control points",
        "metadata": metadata}, indent=2), encoding="utf-8")
    return path
