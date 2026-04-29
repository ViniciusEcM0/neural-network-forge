"""
Regression example.

The network learns:

y = 2*x1 + 3*x2 + 1

This is a simple regression problem.
A linear output with MSE loss is appropriate here.
"""

from neural_forge.network import Network
from datasets.regression import dataset, regression_expected_value


def run():
    network = Network(
        layers=[2, 1],
        internal_act="linear",
        last_act="linear",
        loss="mse",
    )

    lr = 0.01
    epochs = 1000

    network.train(
        dataset=dataset,
        lr=lr,
        epochs=epochs,
        log=True,
        log_interval=100,
    )

    tests = [
        [5, 1],
        [2, 3],
        [10, 0],
        [0, 4],
        [7, 2],
    ]

    for inputs in tests:
        expected = regression_expected_value(inputs)
        pred = network.predict(inputs)[0]

        print(f"x = {inputs}")
        print(f"Esperado: {expected}")
        print(f"Previsto: {pred}")
        print()


if __name__ == "__main__":
    run()