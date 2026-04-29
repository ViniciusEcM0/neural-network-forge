"""
XOR dataset.

Rule:
- Returns 0 when both inputs are equal.
- Returns 1 when inputs are different.

This is a classic non-linear problem.
A purely linear network cannot solve XOR correctly.
"""

dataset = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]