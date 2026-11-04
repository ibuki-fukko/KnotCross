"""Single-factor follow-up: correct oversampled-positive loss weighting."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments.meeting_followup import (
    OUT, SEED, fit_variant, make_query_data, load_pair_dataset,
    choose_decode, write_json,
)


def main():
    train_set = load_pair_dataset("train", ROOT/"data/meeting_analytic_train.npz")
    valid_set = load_pair_dataset("validation", ROOT/"data/meeting_analytic_validation.npz")
    train_queries = make_query_data(train_set, SEED+1)
    valid_queries = make_query_data(valid_set, SEED+2)
    model = fit_variant("direct_raw_plus_grid_prior_corrected", train_queries, valid_queries,
                        input_residual=True, positive_weight=0.05)
    thresholds = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                  0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95)
    selection = choose_decode(model, valid_set, "local", thresholds)
    write_json(OUT/"direct_raw_plus_grid_prior_corrected"/"validation_decode_selection.json", selection)
    print(json.dumps({"positive_query_fraction": float(train_queries["target"][:,2].mean()),
        "positive_class_weight": 0.05, "validation_f1": selection["validation_full"]["f1"],
        "validation_precision": selection["validation_full"]["precision"],
        "validation_recall": selection["validation_full"]["recall"],
        "grid": selection["grid"], "threshold": selection["threshold"],
        "suppression_radius": selection["suppression_radius"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
