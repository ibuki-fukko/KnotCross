"""Construct controlled NURBS pairs whose intersection parameters are known."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.geometry.nurbs import curve_points

def make_pair(parameter_a, parameter_b, crossing_angle, offset, scale=1.0):
    base_parameters = np.linspace(0.0, 1.0, 8)
    points_a = np.column_stack((2.0 * base_parameters - 1.0, 0.55 * np.sin(2.0 * np.pi * base_parameters)))
    curve_a = points_a.copy()
    tangent = np.array([np.cos(crossing_angle), np.sin(crossing_angle)])
    normal = np.array([-tangent[1], tangent[0]])
    local_points = np.column_stack((2.0 * base_parameters - 1.0, 0.45 * np.sin(2.0 * np.pi * base_parameters + 0.6)))
    curve_b = local_points.copy()
    rotation = np.array([[tangent[0], -tangent[1]], [tangent[1], tangent[0]]])
    curve_b = curve_b @ rotation.T
    center = curve_points(curve_a, np.asarray([parameter_a]))[0]
    current = curve_points(curve_b, np.asarray([parameter_b]))[0]
    curve_b = curve_b - current + center + normal * offset
    return curve_a * scale, curve_b * scale


def generate(output="data/known_intersections.npz", seed=20261025):
    rng = np.random.default_rng(seed)
    curves_a = []
    curves_b = []
    labels = []
    counts = []
    cases = []
    for index in range(160):
        parameter_a = float(rng.uniform(0.12, 0.88))
        parameter_b = float(rng.uniform(0.12, 0.88))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        curve_a, curve_b = make_pair(parameter_a, parameter_b, angle, offset=0.0)
        curves_a.append(curve_a)
        curves_b.append(curve_b)
        labels.append(np.asarray([[parameter_a, parameter_b], [0.0, 0.0], [0.0, 0.0]]))
        counts.append(1)
        cases.append({"type": "known_crossing", "u": parameter_a, "v": parameter_b})
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_curve_a=np.asarray(curves_a[:120]),
        train_curve_b=np.asarray(curves_b[:120]),
        train_labels=np.asarray(labels[:120]),
        train_counts=np.asarray(counts[:120]),
        validation_curve_a=np.asarray(curves_a[120:140]),
        validation_curve_b=np.asarray(curves_b[120:140]),
        validation_labels=np.asarray(labels[120:140]),
        validation_counts=np.asarray(counts[120:140]),
        test_curve_a=np.asarray(curves_a[140:]),
        test_curve_b=np.asarray(curves_b[140:]),
        test_labels=np.asarray(labels[140:]),
        test_counts=np.asarray(counts[140:]),
    )
    output_path.with_suffix(".json").write_text(json.dumps({"seed": seed, "cases": cases}, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    generate()
