import random


def generate_circle_dataset(amount=500, radius=0.6):
    dataset = []

    for _ in range(amount):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        distance_squared = x ** 2 + y ** 2

        if distance_squared <= radius ** 2:
            label = 1
        else:
            label = 0

        dataset.append(([x, y], [label]))

    return dataset