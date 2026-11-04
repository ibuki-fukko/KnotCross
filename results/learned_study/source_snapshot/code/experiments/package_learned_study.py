"""Save the study's source, environment, configuration and measurement tables."""

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from code.experiments.controlled_study import STUDY, BASE, config_from, digest, read_json, write_json


def main():
    stages = read_json(STUDY/"stages.json")
    rows = []
    change_audit = []
    for stage, record in stages.items():
        parent = config_from(record["parent"]) if record["parent"] else dict(BASE)
        for path in record["candidates"]:
            config = config_from(path)
            changed = [key for key in BASE if config[key] != parent[key]]
            assert not changed or changed == [record["factor"]], (stage, changed)
            report = read_json(Path(path)/"metrics.json")
            change_audit.append({"stage": stage, "path": path, "changed_fields": changed,
                "selected": path == record["selected"]})
            rows.append({"stage": stage, "run": Path(path).name, "factor": record["factor"],
                "value": config[record["factor"]], "selected_in_stage": path == record["selected"],
                "validation_f1": report["validation_metrics"]["f1"],
                "reference_final_loss": report["validation_reference_final_loss"],
                "own_best_loss": report["best_validation_loss"], "best_epoch": report["best_epoch"],
                "probability_threshold": report["selected_probability_threshold"]})
    with (STUDY/"single_factor_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(STUDY/"single_factor_audit.json", {"all_stages_one_factor": True, "records": change_audit})
    selected = Path(read_json(STUDY/"selection.json")["selected"])
    configs = ROOT/"configs/learned_study"
    write_json(configs/"baseline.json", dict(BASE))
    write_json(configs/"selected.json", config_from(selected))
    write_json(configs/"seed_runs.json", read_json(STUDY/"seed_runs.json"))
    snapshot = STUDY/"source_snapshot"
    paths = list((ROOT/"code").rglob("*.py"))+list((ROOT/"code").rglob("*.ps1"))
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        target = snapshot/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in paths if "__pycache__" not in path.parts}
    write_json(STUDY/"source_manifest.json", manifest)
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    write_json(STUDY/"environment.json", {"python": sys.version, "python_executable": sys.executable,
        "numpy": np.__version__, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "blas_threads": 1,
        "source_sha256": manifest, "main_data_sha256": digest(ROOT/"data/experiment_large.npz"),
        "fresh_holdout_sha256": digest(ROOT/"data/learned_study_locked_test.npz"),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "git_status": git_status.stdout, "no_automatic_commit": True})
    with np.load(ROOT/"data/experiment_large.npz") as archive:
        sample_sets = {}
        for split in ("train", "validation", "test"):
            sample_sets[split] = {hashlib.sha256(np.concatenate([curve_a.ravel(), curve_b.ravel()]).tobytes()).hexdigest()
                for curve_a, curve_b in zip(archive[f"{split}_curve_a"], archive[f"{split}_curve_b"])}
        overlap = {first+"_"+second: len(sample_sets[first]&sample_sets[second])
            for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    assert not any(overlap.values())
    write_json(STUDY/"split_overlap.json", {"exact_pair_overlap": overlap,
        "limitation": "exact hashes do not prove disjoint shape families or distributions"})
    (configs/"requirements_training.txt").write_text("numpy==2.3.2\n", encoding="utf-8")
    print(json.dumps({"stage_count": len(stages), "table_rows": len(rows), "source_files": len(manifest),
        "single_factor_audit_passed": True}, indent=2))


if __name__ == "__main__":
    main()
