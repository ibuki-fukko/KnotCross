"""Aggregate strict refined evaluations across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(records, key):
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values": values.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=[
        "results/autoresearch/refined_candidate_v7/metrics.json",
        "results/autoresearch/refined_seed_7/metrics.json",
        "results/autoresearch/refined_seed_19/metrics.json",
    ])
    parser.add_argument("--output", default="results/autoresearch/refined_seed_summary.json")
    args = parser.parse_args()
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    metrics = [record["test_metrics_filtered"] for record in records]
    summary = {
        "inputs": args.inputs,
        "count": len(records),
        "test_filtered": {metric: summarize(metrics, metric) for metric in ["precision", "recall", "f1", "mean_parameter_error", "mean_geometry_residual"]},
        "test_raw": {metric: summarize([record["test_metrics_raw"] for record in records], metric) for metric in ["precision", "recall", "f1", "mean_parameter_error", "mean_geometry_residual"]},
        "test_refined_without_filter": {metric: summarize([record["test_metrics_refined"] for record in records], metric) for metric in ["precision", "recall", "f1", "mean_parameter_error", "mean_geometry_residual"]},
        "selected_thresholds": [{"probability": record["selected_probability_threshold"], "geometry": record["selected_geometry_threshold"]} for record in records],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
