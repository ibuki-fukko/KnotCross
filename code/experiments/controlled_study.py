"""Controlled validation-only ablations and separately invoked final evaluation."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import code.experiments.candidate_model as candidate_module
from code.evaluation.set_metrics import match_metrics
from code.experiments.learned_correction import (
    LearnedCorrectionModel, build_split, decode_cached, predict_cached,
    state_loss_and_gradient, train, nurbs_points_batch, evaluate,
)

STUDY = Path(os.environ.get("NURBS_STUDY_DIR", ROOT / "results/learned_study"))
DATA = ROOT / "data/experiment_large.npz"
BASE = dict(data=str(DATA), epochs=20, batch_size=256, learning_rate=0.0005,
            seed=20261026, grid_size=48, max_candidates=8, local_slack=0.015,
            steps=4, step_scale=0.08, geometry_weight=1.0, regression_weight=4.0,
            positive_weight=7.0, deep_supervision=0.0, evaluate_test=False, reference_coordinates=False, dynamic_patches=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepared():
    candidate_module.GRID_SIZE = 48
    candidate_module.MAX_CANDIDATES = 8
    cache_path = STUDY / "train_validation_features.npz"
    source_hash = digest(DATA)
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists() and read_json(metadata_path)["data_sha256"] == source_hash:
        with np.load(cache_path) as archive:
            return tuple({key.split("__", 1)[1]: (archive[key].item() if archive[key].ndim == 0 else archive[key])
                          for key in archive.files if key.startswith(split+"__")}
                         for split in ("train", "validation"))
    with np.load(DATA) as archive:
        result = tuple(build_split(archive, split, 48, 8, 0.015) for split in ("train", "validation"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{split+"__"+key: value for split, values in zip(("train", "validation"), result)
                                     for key, value in values.items()})
    write_json(metadata_path, {"data_sha256": source_hash, "grid_size": 48, "max_candidates": 8,
                               "local_slack": 0.015, "positive_assignment_radius": 0.085})
    return result


def calibrated_metrics(model, validation, steps, output, old_threshold=None):
    current, probability = predict_cached(model, validation, steps)
    sweep = []
    for threshold in np.round(np.arange(0.05, 0.991, 0.01), 2):
        prediction = decode_cached(current, probability, validation["groups"], len(validation["counts"]), threshold)
        metrics = match_metrics(prediction, validation["labels"], validation["counts"])
        sweep.append({"threshold": float(threshold), "f1": metrics["f1"], "metrics": metrics})
    best = max(sweep, key=lambda item: (item["f1"], -item["threshold"]))
    result = {"validation_metrics": best["metrics"], "selected_probability_threshold": best["threshold"],
              "validation_threshold_results": [{"threshold": item["threshold"], "f1": item["f1"]} for item in sweep]}
    if old_threshold is not None:
        prediction = decode_cached(current, probability, validation["groups"], len(validation["counts"]), old_threshold)
        result["same_old_threshold_metrics"] = match_metrics(prediction, validation["labels"], validation["counts"])
    np.savez_compressed(Path(output)/"validation_outputs.npz", parameters=current, probability=probability,
                        groups=validation["groups"])
    return result


def run_one(name, config, prepared_data):
    directory = STUDY / name
    if (directory/"metrics.json").exists():
        print("Already completed:", name, flush=True)
        return str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"requested_config.json", config)
    started = time.perf_counter()
    report = train(output=str(directory), prepared_data=prepared_data, **config)
    report["training_and_validation_seconds"] = time.perf_counter()-started
    report["data_sha256"] = digest(DATA)
    report["code_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in [
        ROOT/"code/experiments/learned_correction.py", ROOT/"code/evaluation/set_metrics.py",
        ROOT/"code/geometry/vectorized_basis.py", ROOT/"code/models/mlp_numpy.py"]}
    write_json(directory/"metrics.json", report)
    return str(directory)


def config_from(directory):
    config = read_json(Path(directory)/"requested_config.json")
    return {**BASE, **{key: value for key, value in config.items() if key in BASE}}


def choose(paths):
    def score(path):
        report = read_json(Path(path)/"metrics.json")
        return (report["validation_metrics"]["f1"], -report["validation_reference_final_loss"])
    return max(paths, key=score)


def audit():
    _, validation = prepared()
    records = []
    for name in ("baseline_steps4", "deep_supervision05_steps4", "deep05_steps6"):
        source = ROOT/"results/learned_correction"/name
        output = STUDY/"historical_reassessment"/name
        output.mkdir(parents=True, exist_ok=True)
        config = read_json(source/"config.json")
        model = LearnedCorrectionModel(seed=config["seed"], step_scale=config["step_scale"])
        with np.load(source/"weights.npz") as weights:
            model.load_state(weights)
        original = read_json(source/"metrics.json")
        result = calibrated_metrics(model, validation, config["steps"], output,
                                    original["selected_probability_threshold"])
        result["original_validation_f1_under_old_evaluator"] = original["validation_metrics"]["f1"]
        result["source_weights_sha256"] = digest(source/"weights.npz")
        result["test_was_not_read"] = True
        write_json(output/"metrics.json", result)
        records.append({"name": name, "old_f1": result["original_validation_f1_under_old_evaluator"],
            "corrected_f1_same_threshold": result["same_old_threshold_metrics"]["f1"],
            "recalibrated_f1": result["validation_metrics"]["f1"]})
    write_json(STUDY/"historical_reassessment.json", records)
    print(json.dumps(records, indent=2), flush=True)


def ablate(stage):
    prepared_data = prepared()
    stages = read_json(STUDY/"stages.json") if (STUDY/"stages.json").exists() else {}
    previous = {"steps": "depth", "scale": "steps", "positive": "scale", "geometry": "positive", "duration": "positive",
                "regression": "duration", "reference": "regression", "depth_revisit": "reference",
                "dynamic": "depth_revisit", "dynamic_depth": "dynamic", "dynamic_steps": "dynamic_depth",
                "refresh_depth": "refresh_duration", "refresh_steps": "refresh_depth", "refresh_scale": "refresh_steps",
                "refresh_rate": "refresh_scale", "refresh_regression": "refresh_rate"}
    if stage == "refresh_duration":
        parent = str(STUDY / "dynamic_True")
    else:
        parent = stages[previous[stage]]["selected"] if stage != "depth" else None
    config = config_from(parent) if parent else dict(BASE)
    options = {"depth": ("deep_supervision", [0.0, 0.5]), "steps": ("steps", [4, 6, 8]),
               "scale": ("step_scale", [0.04, 0.08, 0.12, 0.20]),
               "positive": ("positive_weight", [3.0, 5.0, 7.0]),
               "geometry": ("geometry_weight", [0.0, 1.0]), "duration": ("epochs", [20, 40]),
               "regression": ("regression_weight", [4.0, 40.0, 200.0]),
               "reference": ("reference_coordinates", [False, True]),
               "depth_revisit": ("deep_supervision", [0.0, 0.5]),
               "dynamic": ("dynamic_patches", [False, True]),
               "dynamic_depth": ("deep_supervision", [0.0, 0.5]),
               "dynamic_steps": ("steps", [2, 4, 6]),
               "refresh_duration": ("epochs", [20, 40, 80]),
               "refresh_depth": ("deep_supervision", [0.0, 0.5]),
               "refresh_steps": ("steps", [2, 4, 6]),
               "refresh_scale": ("step_scale", [0.01, 0.02, 0.04, 0.08, 0.12]),
               "refresh_rate": ("learning_rate", [0.0001, 0.00025, 0.0005]),
               "refresh_regression": ("regression_weight", [4.0, 40.0])}
    factor, values = options[stage]
    paths = []
    jobs = []
    for value in values:
        trial = {**config, factor: value}
        if parent is not None and trial == config:
            paths.append(parent)
        else:
            jobs.append((stage+"_"+str(value).replace(".", "p"), trial))
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_one, name, trial, prepared_data) for name, trial in jobs]
        paths.extend(future.result() for future in futures)
    selected = choose(paths)
    stages[stage] = {"factor": factor, "parent": parent, "candidates": paths, "selected": selected,
                     "selection": "validation F1 first, reference final loss breaks ties; no test access"}
    write_json(STUDY/"stages.json", stages)
    print("STAGE", stage, "SELECTED", selected, flush=True)


def seeds():
    data = prepared()
    stages = read_json(STUDY/"stages.json")
    selected = choose([stages["dynamic_steps"]["selected"], stages["refresh_regression"]["selected"]])
    write_json(STUDY/"selection.json", {"selected": selected,
        "compared": [stages["dynamic_steps"]["selected"], stages["refresh_regression"]["selected"]],
        "criterion": "validation F1 at tolerance 0.035, then reference final loss",
        "selection_before_test": True})
    final_config = config_from(selected)
    paths = {"baseline": [], "selected": []}
    for family, config in (("baseline", dict(BASE)), ("selected", final_config)):
        for seed in (20261026, 7, 19):
            trial = {**config, "seed": seed}
            if family == "baseline" and seed == 20261026:
                paths[family].append(stages["depth"]["candidates"][0])
            elif family == "selected" and seed == 20261026:
                paths[family].append(selected)
            else:
                paths[family].append(run_one(f"seeds/{family}_{seed}", trial, data))
    write_json(STUDY/"seed_runs.json", paths)
    print(json.dumps(paths, indent=2), flush=True)


def finalize():
    marker = STUDY/"final_evaluation.json"
    if marker.exists() and "fresh_holdout" in read_json(marker):
        print("Final evaluation already exists; retained without re-selection.")
        return
    paths = read_json(STUDY/"seed_runs.json")
    lock = {"families": paths, "thresholds": {path: read_json(Path(path)/"metrics.json")["selected_probability_threshold"]
            for values in paths.values() for path in values}, "test_policy": "historically observed fixed test; descriptive only",
            "locked_at_unix": time.time(), "data_sha256": digest(DATA)}
    write_json(STUDY/"final_lock.json", lock)
    fresh = ROOT/"data/learned_study_locked_test.npz"
    from code.data.locked_reference import create
    create(fresh, count=400, seed=98711351)
    with np.load(DATA) as original, np.load(fresh) as holdout:
        seen = {hashlib.sha256(np.concatenate([curve_a.ravel(), curve_b.ravel()]).tobytes()).hexdigest()
                for curve_a, curve_b in zip(original["train_curve_a"], original["train_curve_b"])}
        holdout_hashes = [hashlib.sha256(np.concatenate([curve_a.ravel(), curve_b.ravel()]).tobytes()).hexdigest()
                         for curve_a, curve_b in zip(holdout["test_curve_a"], holdout["test_curve_b"])]
        overlap = sum(value in seen for value in holdout_hashes)
    if overlap:
        raise RuntimeError("Fresh holdout overlaps training data")
    lock["exact_curve_pair_overlap_with_training"] = overlap
    lock["fresh_holdout_sha256"] = digest(fresh)
    write_json(STUDY/"final_lock.json", lock)
    candidate_module.GRID_SIZE = 48
    candidate_module.MAX_CANDIDATES = 8
    with np.load(DATA) as archive:
        test_data = build_split(archive, "test", 48, 8, 0.015)
    families = {}
    for family, runs in paths.items():
        measurements = []
        for path in runs:
            config = config_from(path)
            model = LearnedCorrectionModel(seed=config["seed"], step_scale=config["step_scale"],
                                           reference_coordinates=config.get("reference_coordinates", False), dynamic_patches=config.get("dynamic_patches", False))
            with np.load(Path(path)/"weights.npz") as weights:
                model.load_state(weights)
            threshold = lock["thresholds"][path]
            predicted, metrics = evaluate(model, test_data, "test", 48, 8, 0.015, threshold, config["steps"])
            all_residuals = []
            all_predicted = []
            for curve_a, curve_b, prediction in zip(test_data["curve_pairs_a"], test_data["curve_pairs_b"], predicted):
                if not prediction:
                    continue
                parameters = np.asarray(prediction)
                point_a = nurbs_points_batch(np.repeat(curve_a[None], len(parameters), axis=0), parameters[:, 0])
                point_b = nurbs_points_batch(np.repeat(curve_b[None], len(parameters), axis=0), parameters[:, 1])
                all_residuals.extend(np.linalg.norm(point_a-point_b, axis=1).tolist())
            metrics["all_predictions_mean_geometry_residual"] = float(np.mean(all_residuals)) if all_residuals else None
            metrics["all_predictions_p95_geometry_residual"] = float(np.percentile(all_residuals, 95)) if all_residuals else None
            metrics["f1_by_parameter_tolerance"] = {str(tolerance): match_metrics(predicted, test_data["labels"], test_data["counts"], tolerance)["f1"]
                for tolerance in [0.005, 0.01, 0.02, 0.035]}
            metrics["seed"] = config["seed"]
            metrics["threshold"] = threshold
            write_json(Path(path)/"final_test_metrics.json", metrics)
            write_json(Path(path)/"final_test_predictions.json", predicted)
            measurements.append(metrics)
        families[family] = {"runs": measurements,
            "mean_f1": float(np.mean([item["f1"] for item in measurements])),
            "std_f1_population": float(np.std([item["f1"] for item in measurements])),
            "mean_parameter_error": float(np.mean([item["mean_parameter_error"] for item in measurements]))}
    write_json(marker, {"families": families, "test_candidate_coverage": test_data["candidate_truth_coverage"],
                       "test_pairs": len(test_data["counts"]), "test_not_used_for_selection": True,
                       "historical_test_exposure": True})
    print(json.dumps({name: {key: value for key, value in values.items() if key != "runs"} for name, values in families.items()}, indent=2), flush=True)
    with np.load(fresh) as archive:
        fresh_data = build_split(archive, "test", 48, 8, 0.015)
    fresh_families = {}
    for family, runs in paths.items():
        measurements = []
        for path in runs:
            config = config_from(path)
            model = LearnedCorrectionModel(seed=config["seed"], step_scale=config["step_scale"],
                                           reference_coordinates=config.get("reference_coordinates", False), dynamic_patches=config.get("dynamic_patches", False))
            with np.load(Path(path)/"weights.npz") as weights:
                model.load_state(weights)
            predicted, metrics = evaluate(model, fresh_data, "test", 48, 8, 0.015,
                                           lock["thresholds"][path], config["steps"])
            current, probability = predict_cached(model, fresh_data, config["steps"])
            zero_predictions = decode_cached(fresh_data["features"][:, :2], probability, fresh_data["groups"],
                                             len(fresh_data["counts"]), lock["thresholds"][path])
            metrics["zero_update_f1_same_threshold"] = match_metrics(zero_predictions, fresh_data["labels"], fresh_data["counts"])["f1"]
            metrics["f1_by_parameter_tolerance"] = {str(tolerance): match_metrics(predicted, fresh_data["labels"], fresh_data["counts"], tolerance)["f1"]
                for tolerance in [0.005, 0.01, 0.02, 0.035]}
            residuals = []
            for curve_a, curve_b, prediction in zip(fresh_data["curve_pairs_a"], fresh_data["curve_pairs_b"], predicted):
                if prediction:
                    parameters = np.asarray(prediction)
                    point_a = nurbs_points_batch(np.repeat(curve_a[None], len(parameters), axis=0), parameters[:, 0])
                    point_b = nurbs_points_batch(np.repeat(curve_b[None], len(parameters), axis=0), parameters[:, 1])
                    residuals.extend(np.linalg.norm(point_a-point_b, axis=1).tolist())
            metrics["all_predictions_mean_geometry_residual"] = float(np.mean(residuals)) if residuals else None
            metrics["all_predictions_p95_geometry_residual"] = float(np.percentile(residuals, 95)) if residuals else None
            metrics["seed"] = config["seed"]
            write_json(Path(path)/"fresh_holdout_metrics.json", metrics)
            write_json(Path(path)/"fresh_holdout_predictions.json", predicted)
            measurements.append(metrics)
        fresh_families[family] = {"runs": measurements,
            "mean_f1": float(np.mean([item["f1"] for item in measurements])),
            "std_f1_population": float(np.std([item["f1"] for item in measurements])),
            "mean_parameter_error": float(np.mean([item["mean_parameter_error"] for item in measurements]))}
    final_report = read_json(marker)
    final_report["fresh_holdout"] = {"families": fresh_families, "pairs": len(fresh_data["counts"]),
                                     "candidate_coverage": fresh_data["candidate_truth_coverage"], "data_sha256": digest(fresh)}
    write_json(marker, final_report)
    print("FRESH HOLDOUT", json.dumps({name: {key: value for key, value in values.items() if key != "runs"}
                                       for name, values in fresh_families.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["audit", "depth", "steps", "scale", "positive", "geometry", "duration", "regression", "reference", "depth_revisit", "dynamic", "dynamic_depth", "dynamic_steps", "refresh_duration", "refresh_depth", "refresh_steps", "refresh_scale", "refresh_rate", "refresh_regression", "seeds", "final"], required=True)
    arguments = parser.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    if not (STUDY/"protocol.json").exists():
        write_json(STUDY/"protocol.json", {"base": BASE, "python": sys.version, "numpy": np.__version__,
            "platform": platform.platform(), "executable": sys.executable, "cpu_threads": 1,
            "parameter_matching_tolerance": 0.035, "suppression_radius": 0.025,
            "threshold_search": [0.05, 0.99, 0.01], "checkpoint_selection": "minimum own validation objective",
            "between_configuration_selection": "maximum corrected validation F1, then reference final loss",
            "test": "read only during final stage; already exposed by historical runs",
            "controller": "project runner; skill controller absent from sparse installation; no fabricated controller history"})
    if arguments.stage == "audit":
        audit()
    elif arguments.stage == "seeds":
        seeds()
    elif arguments.stage == "final":
        finalize()
    else:
        ablate(arguments.stage)


if __name__ == "__main__":
    main()
