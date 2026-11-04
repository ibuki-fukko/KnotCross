"""Threshold-constrained maximum-cardinality matching, evaluator version 2."""

import numpy as np


VERSION = "maximum_cardinality_v2"


def matching(predicted, truth, tolerance=0.035):
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    states = {0: (0.0, [])}
    for pred_index, point in enumerate(predicted):
        updated = dict(states)
        for mask, (cost, pairs) in states.items():
            for truth_index, target in enumerate(truth):
                error = float(np.linalg.norm(point - target))
                if mask & (1 << truth_index) or error > tolerance:
                    continue
                new_mask = mask | (1 << truth_index)
                new_cost = cost + error * error
                if new_mask not in updated or new_cost < updated[new_mask][0]:
                    updated[new_mask] = (new_cost, pairs + [(pred_index, truth_index)])
        states = updated
    best_mask = max(states, key=lambda mask: (mask.bit_count(), -states[mask][0]))
    return states[best_mask][1]


def match_metrics(predicted_sets, labels, counts, tolerance=0.035):
    true_positive = false_positive = false_negative = duplicates = 0
    errors = []
    per_count = {}
    for predicted, label, count in zip(predicted_sets, labels, counts):
        truth = label[:int(count)]
        pairs = matching(predicted, truth, tolerance)
        matched = len(pairs)
        true_positive += matched
        false_positive += len(predicted) - matched
        false_negative += int(count) - matched
        errors.extend(float(np.linalg.norm(np.asarray(predicted[pred_index]) - truth[truth_index]))
                      for pred_index, truth_index in pairs)
        duplicates += len(predicted) - len({tuple(np.round(point, 3)) for point in predicted})
        group = per_count.setdefault(str(int(count)), {"pairs": 0, "tp": 0, "fp": 0, "fn": 0})
        group["pairs"] += 1
        group["tp"] += matched
        group["fp"] += len(predicted) - matched
        group["fn"] += int(count) - matched
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    empty = [len(predicted_sets[index]) > 0 for index in range(len(counts)) if counts[index] == 0]
    for group in per_count.values():
        group["f1"] = 2 * group["tp"] / max(2 * group["tp"] + group["fp"] + group["fn"], 1)
    return {
        "evaluator_version": VERSION, "tolerance": tolerance,
        "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
        "precision": precision, "recall": recall,
        "f1": 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1),
        "mean_parameter_error": float(np.mean(errors)) if errors else None,
        "max_parameter_error": float(np.max(errors)) if errors else None,
        "count_accuracy": float(np.mean([len(predicted) == count for predicted, count in zip(predicted_sets, counts)])),
        "empty_false_positive_rate": float(np.mean(empty)) if empty else None,
        "duplicate_prediction_rate": duplicates / max(sum(map(len, predicted_sets)), 1),
        "predicted_count": float(np.mean([len(item) for item in predicted_sets])),
        "per_intersection_count": per_count,
    }
