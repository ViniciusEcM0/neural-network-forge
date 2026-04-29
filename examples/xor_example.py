"""
XOR example.

This example shows why non-linear activations matter.

Expected behavior:
- A network with only linear layers should get stuck around 50% accuracy.
- A network with ReLU hidden layers should learn XOR.
"""

from neural_forge.network import Network
from datasets.xor import dataset


def run():
    network = Network(
        layers=[2, 4, 1],
        internal_act="relu",
        last_act="sigmoid",
        loss="binary_cross_entropy",
    )

    lr = 0.01
    epochs = 5000

    network.train(
        dataset=dataset,
        lr=lr,
        epochs=epochs,
        log=True,
        log_interval=500,
    )

    accuracy = network.evaluate(dataset)
    print(f"Acurácia treinamento: {accuracy * 100:.2f}%")
    print()

    for inputs, expected in dataset:
        pred = network.predict(inputs)[0]
        predicted_class = 1 if pred >= 0.5 else 0

        print(f"x = {inputs}")
        print(f"Esperado: {expected[0]}")
        print(f"Previsto bruto: {pred}")
        print(f"Classe prevista: {predicted_class}")
        print()


if __name__ == "__main__":
    run()