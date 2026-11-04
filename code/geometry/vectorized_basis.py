"""Vectorized Cox-de Boor basis and one-sided endpoint derivatives."""

import numpy as np

from code.geometry.nurbs import DEGREE, KNOTS


def basis_matrix(parameters, degree=DEGREE, knots=KNOTS):
    values = np.clip(np.asarray(parameters, dtype=float).reshape(-1), knots[degree], knots[-degree-1])
    values = np.minimum(values, np.nextafter(float(knots[-degree-1]), -np.inf))
    level = ((values[:, None] >= knots[:-1]) & (values[:, None] < knots[1:])).astype(float)
    for order in range(1, degree + 1):
        size = len(knots) - order - 1
        left_denominator = knots[order:order+size] - knots[:size]
        right_denominator = knots[order+1:order+size+1] - knots[1:size+1]
        left = np.divide(values[:, None] - knots[:size], left_denominator,
                         out=np.zeros((len(values), size)), where=left_denominator > 0)
        right = np.divide(knots[order+1:order+size+1] - values[:, None], right_denominator,
                          out=np.zeros((len(values), size)), where=right_denominator > 0)
        level = left * level[:, :size] + right * level[:, 1:size+1]
    return level


def basis_derivative_matrix(parameters, degree=DEGREE, knots=KNOTS):
    lower = basis_matrix(parameters, degree - 1, knots)
    size = len(knots) - degree - 1
    left_denominator = knots[degree:degree+size] - knots[:size]
    right_denominator = knots[degree+1:degree+size+1] - knots[1:size+1]
    left = np.divide(degree, left_denominator, out=np.zeros(size), where=left_denominator > 0)
    right = np.divide(degree, right_denominator, out=np.zeros(size), where=right_denominator > 0)
    return lower[:, :size] * left - lower[:, 1:size+1] * right
