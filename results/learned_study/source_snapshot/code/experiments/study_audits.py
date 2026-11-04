"""Diagnostics, inference isolation, exact-case evaluation and reproducibility."""

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.experiments.controlled_study import STUDY, prepared, read_json, write_json, config_from, run_one
from code.experiments.learned_correction import LearnedCorrectionModel, predict_pair, predict_cached, decode_cached
from code.evaluation.set_metrics import match_metrics


def load_model(path):
    config = config_from(path)
    model = LearnedCorrectionModel(seed=config["seed"], step_scale=config["step_scale"],
        reference_coordinates=config["reference_coordinates"], dynamic_patches=config["dynamic_patches"])
    with np.load(Path(path)/"weights.npz") as archive:
        model.load_state(archive)
    return model, config


def diagnose():
    _, validation = prepared()
    records = []
    for directory in sorted(STUDY.glob("*")):
        if not (directory/"requested_config.json").exists() or not (directory/"metrics.json").exists():
            continue
        model, config = load_model(directory)
        current, probability = predict_cached(model, validation, config["steps"])
        report = read_json(directory/"metrics.json")
        threshold = report["selected_probability_threshold"]
        predictions = decode_cached(current, probability, validation["groups"], len(validation["counts"]), threshold)
        unchanged = decode_cached(validation["features"][:, :2], probability, validation["groups"], len(validation["counts"]), threshold)
        positive = validation["target"][:, 2] > 0.5
        target = validation["target"][positive, :2]
        records.append({"name": directory.name, "config": config,
            "f1": report["validation_metrics"]["f1"], "reference_loss": report["validation_reference_final_loss"],
            "initial_positive_parameter_error": float(np.mean(np.linalg.norm(validation["features"][positive, :2]-target, axis=1))),
            "corrected_positive_parameter_error": float(np.mean(np.linalg.norm(current[positive]-target, axis=1))),
            "strict_f1": match_metrics(predictions, validation["labels"], validation["counts"], 0.01)["f1"],
            "zero_update_f1_same_scores": match_metrics(unchanged, validation["labels"], validation["counts"])["f1"]})
    write_json(STUDY/"localization_diagnosis.json", records)
    return records


def isolation():
    _, validation = prepared()
    results = []
    directories = [STUDY/"depth_0p0", STUDY/"dynamic_True"]
    if (STUDY/"selection.json").exists():
        directories.append(Path(read_json(STUDY/"selection.json")["selected"]))
    indices = [int(np.flatnonzero(validation["counts"] == count)[0]) for count in range(4)]
    for directory in directories:
        model, config = load_model(directory)
        for index in indices:
            arguments = (model, validation["curve_pairs_a"][index], validation["curve_pairs_b"][index], 48, 8, 0.015, 0.05, config["steps"])
            expected = predict_pair(*arguments)
            with patch("code.geometry.nurbs.refine_intersection", side_effect=AssertionError("Forbidden Newton call")), \
                 patch("numpy.linalg.solve", side_effect=AssertionError("Forbidden inference solve")), \
                 patch("numpy.linalg.inv", side_effect=AssertionError("Forbidden inference inverse")), \
                 patch("code.experiments.learned_correction.geometric_terms", side_effect=AssertionError("Forbidden residual verification")):
                actual = predict_pair(*arguments)
            assert expected == actual
        results.append({"model": directory.name, "newton_and_linear_solve_disabled": True,
            "training_geometry_function_disabled": True, "predictions_unchanged": True,
            "validation_pair_indices": indices,
            "final_acceptance": "network probability and parameter-domain duplicate suppression only"})
    write_json(STUDY/"inference_isolation.json", results)
    return results


def reproduce():
    original = Path(read_json(STUDY/"selection.json")["selected"])
    repeated = Path(run_one("reproducibility_repeat", config_from(original), prepared()))
    with np.load(original/"weights.npz") as first, np.load(repeated/"weights.npz") as second:
        differences = {key: float(np.max(np.abs(first[key]-second[key]))) for key in first.files}
    with np.load(original/"validation_outputs.npz") as first, np.load(repeated/"validation_outputs.npz") as second:
        output_difference = float(np.max(np.abs(first["parameters"]-second["parameters"])))
    records = {"original": str(original), "repeat": str(repeated), "weight_max_absolute_difference": max(differences.values()),
        "parameter_output_max_absolute_difference": output_difference,
        "f1_original": read_json(original/"metrics.json")["validation_metrics"]["f1"],
        "f1_repeat": read_json(repeated/"metrics.json")["validation_metrics"]["f1"],
        "scope": "same machine, Python, NumPy, one BLAS thread and seed"}
    assert max(differences.values()) < 1e-12 and output_difference < 1e-12
    write_json(STUDY/"reproducibility.json", records)
    return records


def analytic():
    if not (STUDY/"final_evaluation.json").exists():
        raise RuntimeError("Analytic evaluation is final-only, after configuration lock")
    from code.data.analytic_cases import create
    path = create(ROOT/"data/learned_study_analytic.npz")
    archive = np.load(path)
    metadata = read_json(path.with_suffix(".json"))["metadata"]
    families = read_json(STUDY/"seed_runs.json")
    lock = read_json(STUDY/"final_lock.json")
    records = {}
    for family, runs in families.items():
        family_records = []
        for directory in runs:
            model, config = load_model(directory)
            threshold = lock["thresholds"][directory]
            predictions = [predict_pair(model, curve_a, curve_b, 48, 8, 0.015, threshold, config["steps"])
                for curve_a, curve_b in zip(archive["curve_a"], archive["curve_b"])]
            categories = {}
            for category in sorted({item["category"] for item in metadata}):
                indices = [index for index, item in enumerate(metadata) if item["category"] == category]
                categories[category] = match_metrics([predictions[index] for index in indices],
                    archive["labels"][indices], archive["counts"][indices])
            family_records.append({"seed": config["seed"], "categories": categories})
            write_json(Path(directory)/"analytic_predictions.json", predictions)
        records[family] = family_records
    write_json(STUDY/"analytic_evaluation.json", {"families": records, "data": str(path),
        "not_used_for_selection": True, "all_roots_analytically_known": True})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["diagnose", "isolation", "reproduce", "analytic"], required=True)
    arguments = parser.parse_args()
    result = {"diagnose": diagnose, "isolation": isolation, "reproduce": reproduce, "analytic": analytic}[arguments.action]()
    print(json.dumps(result, indent=2))
