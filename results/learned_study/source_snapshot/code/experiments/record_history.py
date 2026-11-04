from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()
history_path = ROOT / "results" / "autoresearch" / "experiment_history.json"
history = json.loads(history_path.read_text(encoding="utf-8"))
existing = {item.get("experiment") for item in history}
new_items = [
    {
        "experiment": "candidate_v7_geometric_refine_3_filtered",
        "f1": 0.969318181818182,
        "precision": 0.9918604651162791,
        "recall": 0.9477777777777778,
        "validation_loss": 0.13899503076389216,
        "validation_f1": 0.9779286926994907,
        "parameter_error": 2.33338387842919e-05,
        "geometry_residual": 3.995998830170805e-05,
        "probability_threshold": 0.35,
        "geometry_threshold": 0.0055,
        "refine_steps": 3,
    },
    {
        "experiment": "candidate_v7_geometric_refine_8_filtered",
        "f1": 0.9755264655663062,
        "precision": 1.0,
        "recall": 0.9522222222222222,
        "validation_loss": 0.13899503076389216,
        "validation_f1": 0.9821882951653944,
        "parameter_error": 1.921984151321906e-11,
        "geometry_residual": 4.493322827287381e-12,
        "probability_threshold": 0.35,
        "geometry_threshold": 0.0005,
        "refine_steps": 8,
    },
    {
        "experiment": "candidate_v7_geometric_refine_8_three_seed_mean",
        "f1": 0.9770769377955949,
        "precision": 1.0,
        "recall": 0.9551851851851852,
        "parameter_error": 6.507547728774736e-12,
        "geometry_residual": 2.7871274303153053e-12,
        "f1_std": 0.0017749921070452995,
        "refine_steps": 8,
        "seed_count": 3,
    },
]
for item in new_items:
    if item["experiment"] not in existing:
        history.append(item)
history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
print(history_path)
