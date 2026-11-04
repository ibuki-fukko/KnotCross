"""Generate boundary-case curve pairs for separate evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.geometry.nurbs import curve_points


def generate(output="data/boundary_cases.npz", seed=20261025):
    rng = np.random.default_rng(seed)
    parameters = np.linspace(0.0, 1.0, 8)
    base = np.column_stack((2.0 * parameters - 1.0, 0.45 * np.sin(2.0 * np.pi * parameters)))
    curves_a = []
    curves_b = []
    labels = []
    counts = []
    metadata = []
    for case_type in ["near", "endpoint", "tangent"]:
        for index in range(40):
            curve_a = base + rng.normal(0.0, 0.03, base.shape)
            curve_b = base.copy()
            if case_type == "near":
                curve_b = curve_b + np.array([0.0, 0.04 + 0.02 * rng.random()])
                label = np.zeros((3, 2), dtype=np.float64)
                count = 0
            elif case_type == "endpoint":
                target_u = 0.02
                target_v = float(rng.uniform(0.1, 0.9))
                point = curve_points(curve_a, np.asarray([target_u]))[0]
                current = curve_points(curve_b, np.asarray([target_v]))[0]
                curve_b += point - current
                label = np.zeros((3, 2), dtype=np.float64)
                label[0] = [target_u, target_v]
                count = 1
            else:
                target_u = float(rng.uniform(0.15, 0.85))
                target_v = float(rng.uniform(0.15, 0.85))
                point = curve_points(curve_a, np.asarray([target_u]))[0]
                current = curve_points(curve_b, np.asarray([target_v]))[0]
                curve_b += point - current
                label = np.zeros((3, 2), dtype=np.float64)
                label[0] = [target_u, target_v]
                count = 1
            curves_a.append(curve_a)
            curves_b.append(curve_b)
            labels.append(label)
            counts.append(count)
            metadata.append({"case_type": case_type, "index": index})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, curve_a=np.asarray(curves_a), curve_b=np.asarray(curves_b), labels=np.asarray(labels), counts=np.asarray(counts))
    path.with_suffix(".json").write_text(json.dumps({"seed": seed, "metadata": metadata}, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    generate()
