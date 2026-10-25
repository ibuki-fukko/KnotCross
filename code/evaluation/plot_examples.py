"""Create simple PNG examples without requiring matplotlib."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.geometry.nurbs import curve_points
from code.models.mlp_numpy import sigmoid
from code.training.train_experiments import build_features, create_model
from code.evaluation.evaluate_experiments import extract_slot_sets, load_state, neural_predictions


def draw_example(curve_a, curve_b, truth, predicted, output: Path, title: str) -> None:
    image = Image.new("RGB", (720, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 15), title, fill="black")
    points_a = curve_points(curve_a, np.linspace(0.0, 1.0, 240))
    points_b = curve_points(curve_b, np.linspace(0.0, 1.0, 240))
    combined = np.concatenate([points_a, points_b], axis=0)
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    scale = max(float(np.max(maximum - minimum)), 1e-8)

    def project(point):
        x = 70 + int(570 * (point[0] - minimum[0]) / scale)
        y = 500 - int(450 * (point[1] - minimum[1]) / scale)
        return x, y

    draw.line([project(point) for point in points_a], fill=(40, 90, 220), width=3)
    draw.line([project(point) for point in points_b], fill=(220, 80, 50), width=3)
    for parameter_a, parameter_b in truth:
        point = 0.5 * (curve_points(curve_a, np.asarray([parameter_a]))[0] + curve_points(curve_b, np.asarray([parameter_b]))[0])
        x, y = project(point)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(20, 150, 50), width=3)
    for parameter_a, parameter_b in predicted:
        point = 0.5 * (curve_points(curve_a, np.asarray([parameter_a]))[0] + curve_points(curve_b, np.asarray([parameter_b]))[0])
        x, y = project(point)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(150, 0, 150), width=3)
    draw.text((20, 525), "blue/red curves; green truth; purple prediction", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/experiment_large.npz")
    parser.add_argument("--models", default="results/large")
    parser.add_argument("--output", default="results/large/examples")
    args = parser.parse_args()
    archive = np.load(args.data)
    curves_a = archive["test_curve_a"]
    curves_b = archive["test_curve_b"]
    labels = archive["test_labels"]
    counts = archive["test_counts"]
    features = build_features(curves_a, curves_b)
    model = create_model("shared", 20261025)
    model.load_state(load_state(Path(args.models) / "shared_full" / "weights.npz"))
    _, slots = neural_predictions("shared_full", model, features)
    predicted = extract_slot_sets(slots, threshold=0.2)
    selected = []
    for count in [0, 1, 2, 3]:
        index = int(np.flatnonzero(counts == count)[0])
        selected.append(index)
    for index in selected:
        draw_example(
            curves_a[index],
            curves_b[index],
            labels[index, : counts[index]],
            predicted[index],
            Path(args.output) / f"test_{index:04d}_count_{counts[index]}.png",
            f"test sample {index}, true intersections={counts[index]}",
        )
    print(f"saved examples to {args.output}")


if __name__ == "__main__":
    main()
