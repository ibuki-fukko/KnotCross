"""Generate synthetic paired NURBS curves and verified intersection labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.geometry.nurbs import N_CONTROL, affine_normalize_pair, find_intersections


def random_curve(rng: np.random.Generator) -> np.ndarray:
    """Create a smooth control polygon with varied curvature and orientation."""
    parameters = np.linspace(0.0, 1.0, N_CONTROL)
    frequency_x = rng.uniform(0.35, 1.25)
    frequency_y = rng.uniform(0.45, 1.75)
    phase_x = rng.uniform(0.0, 2.0 * np.pi)
    phase_y = rng.uniform(0.0, 2.0 * np.pi)
    x = 2.0 * parameters - 1.0
    x += rng.uniform(0.05, 0.55) * np.sin(2.0 * np.pi * frequency_x * parameters + phase_x)
    y = rng.uniform(0.35, 0.95) * np.sin(2.0 * np.pi * frequency_y * parameters + phase_y)
    y += rng.uniform(0.05, 0.35) * np.cos(2.0 * np.pi * (frequency_y + 0.4) * parameters + phase_x)
    noise = rng.normal(0.0, 0.06, size=(N_CONTROL, 2))
    curve = np.column_stack((x, y)) + noise
    rotation = rng.uniform(-np.pi, np.pi)
    rotation_matrix = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
        dtype=np.float64,
    )
    curve = curve @ rotation_matrix.T
    curve *= rng.uniform(0.75, 1.25)
    curve += rng.normal(0.0, 0.08, size=(1, 2))
    return curve


def apply_hard_negative_transform(curve: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make two nearby but usually disjoint curves for a difficult negative case."""
    transformed = curve.copy()
    angle = rng.uniform(-0.08, 0.08)
    matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    transformed = transformed @ matrix.T
    transformed += rng.uniform(0.03, 0.12, size=(1, 2))
    transformed[:, 1] += 0.035
    return transformed


def generate_split(
    rng: np.random.Generator,
    count: int,
    split_name: str,
    max_intersections: int = 3,
    include_hard_negatives: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    curves_a: list[np.ndarray] = []
    curves_b: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    counts: list[int] = []
    metadata: list[dict] = []
    target_counts = [0, 1, 2, 3]
    attempts = 0
    while len(curves_a) < count:
        attempts += 1
        if attempts > count * 300:
            raise RuntimeError(f"Could not generate enough {split_name} samples after {attempts} attempts")
        target = target_counts[len(curves_a) % len(target_counts)]
        curve_a = random_curve(rng)
        curve_b = random_curve(rng)
        hard_negative = include_hard_negatives and target == 0 and rng.random() < 0.35
        if hard_negative:
            curve_b = apply_hard_negative_transform(curve_a, rng)
        curve_a, curve_b = affine_normalize_pair(curve_a, curve_b)
        intersections = find_intersections(curve_a, curve_b, sample_count=121)
        if len(intersections) != target or len(intersections) > max_intersections:
            continue
        curves_a.append(curve_a)
        curves_b.append(curve_b)
        label = np.zeros((max_intersections, 2), dtype=np.float64)
        if intersections:
            label[: len(intersections)] = np.asarray(intersections, dtype=np.float64)
        labels.append(label)
        counts.append(len(intersections))
        metadata.append(
            {
                "split": split_name,
                "sample_index": len(curves_a) - 1,
                "intersection_count": len(intersections),
                "hard_negative": hard_negative,
            }
        )
    return (
        np.asarray(curves_a, dtype=np.float64),
        np.asarray(curves_b, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/nurbs_intersections.npz")
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--validation", type=int, default=400)
    parser.add_argument("--test", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    split_data = {}
    all_metadata: list[dict] = []
    for index, (name, count) in enumerate(
        [("train", args.train), ("validation", args.validation), ("test", args.test)]
    ):
        split = generate_split(np.random.default_rng(args.seed + index * 100003), count, name)
        split_data[name] = split
        all_metadata.extend(split[-1])
        print(f"{name}: {count} paired curves; counts={np.bincount(split[3], minlength=4).tolist()}")
    np.savez_compressed(
        output,
        train_curve_a=split_data["train"][0],
        train_curve_b=split_data["train"][1],
        train_labels=split_data["train"][2],
        train_counts=split_data["train"][3],
        validation_curve_a=split_data["validation"][0],
        validation_curve_b=split_data["validation"][1],
        validation_labels=split_data["validation"][2],
        validation_counts=split_data["validation"][3],
        test_curve_a=split_data["test"][0],
        test_curve_b=split_data["test"][1],
        test_labels=split_data["test"][2],
        test_counts=split_data["test"][3],
    )
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "degree": 3,
                "control_points": N_CONTROL,
                "knot_vector": [0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0],
                "weights": "all ones",
                "label_method": "dense polyline segment crossings followed by Newton refinement and deduplication",
                "metadata": all_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {output}")
    print(f"saved: {metadata_path}")


if __name__ == "__main__":
    main()
