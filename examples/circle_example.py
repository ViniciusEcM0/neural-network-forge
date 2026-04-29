"""
Circle classification example.

The network receives [x, y] and learns whether the point is inside
or outside a circle.

Target:
- 1 = inside the circle
- 0 = outside the circle
"""

from neural_forge.network import Network
from datasets.circle import generate_circle_dataset, circle_expected_class


def run():
    train_dataset = generate_circle_dataset(amount=1000, radius=0.6)
    test_dataset = generate_circle_dataset(amount=200, radius=0.6)

    network = Network(
        layers=[2, 8, 8, 1],
        internal_act="relu",
        last_act="sigmoid",
        loss="binary_cross_entropy",
    )

    lr = 0.001
    epochs = 2000

    network.train(
        dataset=train_dataset,
        lr=lr,
        epochs=epochs,
        log=True,
        log_interval=100,
    )

    train_accuracy = network.evaluate(train_dataset)
    test_accuracy = network.evaluate(test_dataset)

    print(f"Acurácia treinamento: {train_accuracy * 100:.2f}%")
    print(f"Acurácia teste: {test_accuracy * 100:.2f}%")
    print()

    tests = [
        [0, 0],
        [0.2, 0.2],
        [0.5, 0],
        [0.7, 0],
        [1, 1],
        [-0.3, 0.4],
        [-0.8, 0.2],
    ]

    for inputs in tests:
        expected = circle_expected_class(inputs, radius=0.6)
        pred = network.predict(inputs)[0]
        predicted_class = 1 if pred >= 0.5 else 0

        print(f"x = {inputs}")
        print(f"Esperado: {expected}")
        print(f"Previsto bruto: {pred}")
        print(f"Classe prevista: {predicted_class}")
        print()


if __name__ == "__main__":
    run()