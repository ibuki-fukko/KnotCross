"""Lock and evaluate a fresh analytic holdout after the meeting-study models."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments.meeting_followup import (
    OUT, SEED, make_analytic_split, write_json, sha256, DirectGridMLP,
    model_metrics, load_model, GEOMETRIC_TOLERANCES,
)


def analytic_dataset(path):
    archive = np.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["metadata"]
    return {"curve_a": archive["curve_a"], "curve_b": archive["curve_b"],
            "labels": archive["labels"], "counts": archive["counts"],
            "categories": np.asarray([row["category"] for row in metadata], dtype=object)}


def main():
    data_path = ROOT / "data/meeting_analytic_locked_holdout.npz"
    make_analytic_split(data_path, per_type=50, seed=20261107)
    names = ("direct_raw", "direct_raw_plus_grid_input", "direct_raw_plus_global_grid_input")
    locked_models = {}
    for name in names:
        config = json.loads((OUT/name/"config.json").read_text(encoding="utf-8"))
        selection = json.loads((OUT/name/"validation_decode_selection.json").read_text(encoding="utf-8"))
        locked_models[name] = {"config_sha256": sha256(OUT/name/"config.json"),
            "weights_sha256": sha256(OUT/name/"weights.npz"),
            "grid": selection["grid"], "score_threshold": selection["threshold"],
            "parameter_deduplication_radius": selection["suppression_radius"],
            "residual_input_mode": config.get("residual_input_mode", "local" if config.get("residual_patch_in_first_layer") else "none")}
    legacy_path = ROOT / "results/learned_study/refresh_scale_0p04"
    legacy_config = json.loads((legacy_path/"config.json").read_text(encoding="utf-8"))
    lock = {"seed": 20261107, "pairs_per_category": 50, "pair_count": 350,
        "categories": ["single", "multiple", "close_0p02", "close_0p005", "endpoint", "tangent", "near_disjoint"],
        "data_sha256": sha256(data_path), "models": locked_models,
        "legacy_model": {"weights_sha256": sha256(legacy_path/"weights.npz"), "threshold": 0.95,
                         "grid_size": legacy_config["grid_size"], "max_candidates": legacy_config["max_candidates"]},
        "geometric_tolerances_for_quality_audit": GEOMETRIC_TOLERANCES,
        "selection_rule": "all model settings were selected on the separate validation split before this holdout was generated",
        "inference_rule": "no Newton update and no geometric-residual threshold filtering"}
    write_json(OUT/"locked_holdout_lock.json", lock)
    dataset = analytic_dataset(data_path)
    results = {}
    for name in names:
        selection = json.loads((OUT/name/"validation_decode_selection.json").read_text(encoding="utf-8"))
        model = load_model(OUT/name/"config.json")
        mode = locked_models[name]["residual_input_mode"]
        metrics, predictions = model_metrics(model, dataset, selection["grid"], mode,
            selection["threshold"], selection["suppression_radius"])
        results[name] = metrics
        write_json(OUT/name/"locked_holdout_metrics.json", metrics)
        write_json(OUT/name/"locked_holdout_predictions.json",
                   [[[float(u), float(v)] for u,v in points] for points in predictions])

    from code.experiments.meeting_followup import evaluate_legacy_model
    legacy = evaluate_legacy_model(dataset)
    results["legacy_learned_correction"] = legacy["test"]
    write_json(OUT/"legacy_learned_correction"/"locked_holdout_metrics.json", legacy["test"])
    write_json(OUT/"legacy_learned_correction"/"locked_holdout_predictions.json",
               [[[float(u), float(v)] for u,v in points] for points in legacy["predictions"]])
    write_json(OUT/"locked_holdout_results.json", {"lock": lock, "models": results})
    print(json.dumps({name: {"f1_0p035": metric["f1"], "precision": metric["precision"],
        "recall": metric["recall"], "f1_tolerances": metric["f1_by_parameter_tolerance"],
        "categories": metric["per_category"]} for name,metric in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
