"""NURBS geometry utilities used by the reproducible experiments.

The experiment fixes a cubic open-uniform knot vector and positive weights.
The implementation is deliberately self-contained so that the labels and the
training-time geometric residual do not depend on a commercial CAD package.
"""

from __future__ import annotations

import numpy as np


DEGREE = 3
N_CONTROL = 8
KNOTS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float64,
)


def basis_values(u: float, degree: int = DEGREE, knots: np.ndarray = KNOTS) -> np.ndarray:
    """Return all non-rational B-spline basis values at one parameter."""

    n_basis = len(knots) - degree - 1
    value = float(np.clip(u, 0.0, 1.0))
    levels = np.zeros((degree + 1, n_basis + degree), dtype=np.float64)
    if value >= 1.0:
        levels[0, n_basis - 1] = 1.0
    else:
        for index in range(n_basis + degree):
            if knots[index] <= value < knots[index + 1]:
                levels[0, index] = 1.0
    for current_degree in range(1, degree + 1):
        for index in range(n_basis + degree - current_degree):
            left_denominator = knots[index + current_degree] - knots[index]
            right_denominator = knots[index + current_degree + 1] - knots[index + 1]
            left = 0.0
            right = 0.0
            if left_denominator > 0.0:
                left = (value - knots[index]) / left_denominator * levels[current_degree - 1, index]
            if right_denominator > 0.0:
                right = (knots[index + current_degree + 1] - value) / right_denominator * levels[current_degree - 1, index + 1]
            levels[current_degree, index] = left + right
    return levels[degree, :n_basis]


def basis_matrix(parameters: np.ndarray, degree: int = DEGREE, knots: np.ndarray = KNOTS) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    return np.stack([basis_values(parameter, degree, knots) for parameter in parameters], axis=0)


def curve_point(control_points: np.ndarray, parameter: float, weights: np.ndarray | None = None) -> np.ndarray:
    control_points = np.asarray(control_points, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(control_points), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    basis = basis_values(parameter)
    weighted_basis = basis * weights
    denominator = float(np.sum(weighted_basis))
    if denominator <= 1e-14:
        raise ValueError("NURBS denominator is too small")
    return np.sum(weighted_basis[:, None] * control_points, axis=0) / denominator


def curve_points(control_points: np.ndarray, parameters: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    control_points = np.asarray(control_points, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(control_points), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    basis = basis_matrix(parameters)
    weighted_basis = basis * weights[None, :]
    denominator = np.sum(weighted_basis, axis=1, keepdims=True)
    return (weighted_basis @ control_points) / np.maximum(denominator, 1e-14)


def curve_derivative(control_points: np.ndarray, parameter: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Numerical parameter derivative used by the geometric loss and Newton refinement."""

    step = 1e-5
    parameter = float(parameter)
    if parameter <= step:
        return (curve_point(control_points, parameter + step, weights) - curve_point(control_points, parameter, weights)) / step
    if parameter >= 1.0 - step:
        return (curve_point(control_points, parameter, weights) - curve_point(control_points, parameter - step, weights)) / step
    return (curve_point(control_points, parameter + step, weights) - curve_point(control_points, parameter - step, weights)) / (2.0 * step)


def parameter_grid_residual(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    grid_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return parameter grid, dx, dy and distance fields for two curves."""

    parameters = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    points_a = curve_points(curve_a, parameters)
    points_b = curve_points(curve_b, parameters)
    difference = points_a[:, None, :] - points_b[None, :, :]
    dx = difference[:, :, 0]
    dy = difference[:, :, 1]
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    return parameters, dx, dy, distance, difference


def cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def polyline_intersections(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    sample_count: int = 161,
    tolerance: float = 1e-10,
) -> list[tuple[float, float]]:
    """Approximate all ordinary segment crossings and return parameter pairs."""

    parameters = np.linspace(0.0, 1.0, sample_count)
    points_a = curve_points(curve_a, parameters)
    points_b = curve_points(curve_b, parameters)
    candidates: list[tuple[float, float]] = []
    for index_a in range(sample_count - 1):
        first = points_a[index_a]
        direction_a = points_a[index_a + 1] - first
        for index_b in range(sample_count - 1):
            second = points_b[index_b]
            direction_b = points_b[index_b + 1] - second
            denominator = cross_2d(direction_a, direction_b)
            if abs(denominator) < tolerance:
                continue
            difference = second - first
            ratio_a = cross_2d(difference, direction_b) / denominator
            ratio_b = cross_2d(difference, direction_a) / denominator
            if -1e-9 <= ratio_a <= 1.0 + 1e-9 and -1e-9 <= ratio_b <= 1.0 + 1e-9:
                parameter_a = (index_a + np.clip(ratio_a, 0.0, 1.0)) / (sample_count - 1)
                parameter_b = (index_b + np.clip(ratio_b, 0.0, 1.0)) / (sample_count - 1)
                candidates.append((parameter_a, parameter_b))
    return candidates


def refine_intersection(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    initial: tuple[float, float],
    iterations: int = 20,
) -> tuple[float, float, float] | None:
    """Refine a segment crossing with a two-variable Newton solve."""

    parameter_a, parameter_b = map(float, initial)
    for _ in range(iterations):
        point_a = curve_point(curve_a, parameter_a)
        point_b = curve_point(curve_b, parameter_b)
        residual = point_a - point_b
        jacobian = np.column_stack((curve_derivative(curve_a, parameter_a), -curve_derivative(curve_b, parameter_b)))
        if abs(float(np.linalg.det(jacobian))) < 1e-10:
            break
        try:
            update = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            break
        parameter_a = float(np.clip(parameter_a + update[0], 0.0, 1.0))
        parameter_b = float(np.clip(parameter_b + update[1], 0.0, 1.0))
        if float(np.linalg.norm(update)) < 1e-10:
            break
    final_residual = float(np.linalg.norm(curve_point(curve_a, parameter_a) - curve_point(curve_b, parameter_b)))
    if final_residual > 2e-4:
        return None
    return parameter_a, parameter_b, final_residual


def deduplicate_intersections(candidates: list[tuple[float, float, float]], tolerance: float = 2e-3) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for parameter_a, parameter_b, _ in sorted(candidates):
        if all(np.linalg.norm(np.asarray((parameter_a, parameter_b)) - np.asarray(existing)) > tolerance for existing in output):
            output.append((parameter_a, parameter_b))
    return output


def find_intersections(curve_a: np.ndarray, curve_b: np.ndarray, sample_count: int = 161) -> list[tuple[float, float]]:
    refined = []
    for candidate in polyline_intersections(curve_a, curve_b, sample_count):
        result = refine_intersection(curve_a, curve_b, candidate)
        if result is not None:
            refined.append(result)
    return deduplicate_intersections(refined)


def affine_normalize_pair(curve_a: np.ndarray, curve_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([curve_a, curve_b], axis=0)
    center = 0.5 * (combined.min(axis=0) + combined.max(axis=0))
    scale = float(np.max(combined.max(axis=0) - combined.min(axis=0)))
    scale = max(scale, 1e-8)
    return (curve_a - center) * (2.0 / scale), (curve_b - center) * (2.0 / scale)
