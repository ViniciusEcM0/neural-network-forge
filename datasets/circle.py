"""
Circle classification dataset.

The model receives a point [x, y].

Target:
- [1] if the point is inside the circle
- [0] if the point is outside the circle

This is a non-linear classification problem.
"""

import random


def generate_circle_dataset(amount=1000, radius=0.6, min_value=-1, max_value=1):
    dataset = []

    for _ in range(amount):
        x = random.uniform(min_value, max_value)
        y = random.uniform(min_value, max_value)

        distance_squared = x ** 2 + y ** 2
        radius_squared = radius ** 2

        label = 1 if distance_squared <= radius_squared else 0

        dataset.append(([x, y], [label]))

    return dataset


def circle_expected_class(inputs, radius=0.6):
    x, y = inputs
    return 1 if x ** 2 + y ** 2 <= radius ** 2 else 0