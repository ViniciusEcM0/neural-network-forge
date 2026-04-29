"""
Regression dataset.

The model receives [x1, x2] and should learn:

y = 2*x1 + 3*x2 + 1

This is a linear regression problem.
A single linear neuron should be able to learn it.
"""

dataset = [
    ([1, 1], [6]),
    ([2, 1], [8]),
    ([1, 2], [9]),
    ([3, 2], [13]),
    ([4, 1], [12]),
    ([2, 3], [14]),
    ([5, 0], [11]),
    ([0, 5], [16]),
    ([3, 3], [16]),
    ([4, 4], [21]),
]


def regression_expected_value(inputs):
    x1, x2 = inputs
    return 2 * x1 + 3 * x2 + 1