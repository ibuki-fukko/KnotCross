"""Reproducible numerical checks, without reading any experiment test split."""

import json
import itertools
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.geometry.nurbs import basis_matrix as scalar_basis
from code.geometry.vectorized_basis import basis_matrix, basis_derivative_matrix
from code.evaluation.set_metrics import matching, match_metrics
from code.experiments.learned_correction import LearnedCorrectionModel, batch_loss_and_gradient, unroll, supervision_weights, dynamic_patch
from code.experiments.candidate_model import candidate_indices


def main():
    rng = np.random.default_rng(208)
    parameters = np.r_[0.0, 1.0, rng.uniform(0.01, 0.99, 200)]
    basis_error = float(np.max(np.abs(basis_matrix(parameters) - scalar_basis(parameters))))
    interior = parameters[2:]
    numerical = (basis_matrix(interior+1e-6) - basis_matrix(interior-1e-6)) / 2e-6
    derivative_error = float(np.max(np.abs(numerical-basis_derivative_matrix(interior))))
    assert basis_error < 1e-12 and derivative_error < 1e-7
    endpoint_error = float(np.max(np.abs((basis_matrix([1.0])-basis_matrix([1-1e-7]))/1e-7
                                        - basis_derivative_matrix([1.0]))))
    assert endpoint_error < 1e-4
    truth = np.array([[0.2, 0.2], [0.8, 0.8]])
    predictions = [[0.5, 0.5], [0.8, 0.8], [0.2, 0.2]]
    assert len(matching(predictions, truth)) == 2
    assert len(matching([[0.2, 0.2], [0.2, 0.2]], truth)) == 1
    assert matching([], truth) == [] and matching(predictions, []) == []
    ambiguous_truth = np.array([[0.0, 0.0], [0.03, 0.0]])
    assert len(matching([[0.015, 0.0], [0.0, 0.0]], ambiguous_truth, 0.02)) == 2
    for repeat in range(60):
        predicted = rng.uniform(0, 1, (int(rng.integers(0, 6)), 2))
        reference_truth = rng.uniform(0, 1, (int(rng.integers(0, 4)), 2))
        tolerance = 0.4
        best_cardinality, best_cost = 0, 0.0
        for cardinality in range(1, min(len(predicted), len(reference_truth))+1):
            for pred_indices in itertools.combinations(range(len(predicted)), cardinality):
                for truth_indices in itertools.permutations(range(len(reference_truth)), cardinality):
                    errors = np.linalg.norm(predicted[list(pred_indices)]-reference_truth[list(truth_indices)], axis=1)
                    if np.all(errors <= tolerance):
                        cost = float(np.sum(errors**2))
                        if cardinality > best_cardinality or (cardinality == best_cardinality and cost < best_cost):
                            best_cardinality, best_cost = cardinality, cost
        actual = matching(predicted, reference_truth, tolerance)
        cost = sum(float(np.sum((predicted[pred_index]-reference_truth[truth_index])**2)) for pred_index, truth_index in actual)
        assert len(actual) == best_cardinality and abs(cost-best_cost) < 1e-12
    assert supervision_weights(1, 0.5)[0] == 1
    field = rng.uniform(0, 1, (48, 48))
    reference = []
    for row in range(1, 47):
        for column in range(1, 47):
            if field[row, column] <= np.min(field[row-1:row+2, column-1:column+2]) + 0.015:
                reference.append((float(field[row, column]), row, column))
    selected = []
    for value, row, column in sorted(reference):
        if all(abs(row-other_row) > 2 or abs(column-other_column) > 2 for _, other_row, other_column in selected):
            selected.append((value, row, column))
        if len(selected) == 8:
            break
    assert selected == candidate_indices(field, 8, 0.015)
    features = rng.normal(0, 0.1, (6, 77))
    features[:, :2] = rng.uniform(0.35, 0.65, (6, 2))
    target = np.c_[rng.uniform(0.2, 0.8, (6, 2)), [1, 1, 0, 1, 0, 1]]
    curve_a, curve_b = rng.normal(0, 0.2, (2, 6, 8, 2))
    gradient_checks = []
    configurations = [(1, 0.5, 1, False, False), (4, 0, 1, False, False), (4, 0.5, 1, False, False),
                      (6, 0.5, 1, False, False), (4, 0.5, 0, False, False), (4, 0.5, 1, True, False),
                      (4, 0, 1, False, True), (4, 0.5, 1, False, True)]
    for steps, depth, geometry, reference, dynamic in configurations:
        model = LearnedCorrectionModel(seed=34, step_scale=0.03, reference_coordinates=reference, dynamic_patches=dynamic)
        arguments = (features, target, curve_a, curve_b, steps, geometry, 4, 7, depth)
        loss = batch_loss_and_gradient(model, *arguments)[0]
        gradient_errors = []
        for layer_index in [0, 2, 3]:
            layer = model.network.parameter_layers()[layer_index]
            for column in [0, layer.weight.shape[1]-1]:
                row = int(np.argmax(np.abs(layer.grad_weight[:, column])))
                analytic = float(layer.grad_weight[row, column])
                original = layer.weight[row, column]
                layer.weight[row, column] = original + 1e-6
                plus = batch_loss_and_gradient(model, *arguments)[0]
                layer.weight[row, column] = original - 1e-6
                minus = batch_loss_and_gradient(model, *arguments)[0]
                layer.weight[row, column] = original
                numeric = (plus-minus)/2e-6
                error = abs(numeric-analytic)
                assert error < 2e-6, (steps, depth, geometry, analytic, numeric)
                gradient_errors.append(error)
                batch_loss_and_gradient(model, *arguments)
        gradient_checks.append({"steps": steps, "deep_supervision": depth, "geometry_weight": geometry, "reference_coordinates": reference, "dynamic_patches": dynamic,
                                "loss": loss, "max_absolute_gradient_error": max(gradient_errors)})
        before = unroll(model, features, steps, curve_a=curve_a, curve_b=curve_b)[0]
        for repeat in range(4):
            after = unroll(model, features, steps, curve_a=curve_a, curve_b=curve_b)[0]
        np.testing.assert_array_equal(before, after)
        assert len(model.network.parameter_layers()[0].cache) == steps
    result = {"basis_max_absolute_error": basis_error, "derivative_max_absolute_error": derivative_error,
              "one_sided_endpoint_derivative_error": endpoint_error, "gradient_checks": gradient_checks,
              "matching_counterexamples_passed": True, "cache_bound_passed": True,
              "vectorized_candidate_proposal_equivalence": True, "matching_random_bruteforce_checks": 60}
    destination = Path(os.environ.get("NURBS_STUDY_DIR", ROOT/"results/learned_study"))/"checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
