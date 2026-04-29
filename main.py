from neural_forge.layer import Layer
from neural_forge.network import Network
from data import generate_circle_dataset


train_dataset = generate_circle_dataset(1000, 0.6)
test_dataset = generate_circle_dataset(200, 0.6)

dataset = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

lr = 0.01
epochs = 5000

network = Network(
    layers=[2, 4, 1],
    internal_act="relu",
    last_act="linear",
    loss="mse"
)
network.train(dataset, lr, epochs)

train_accuracy = network.evaluate(dataset)

print(f"Acurácia treinamento: {train_accuracy * 100:.2f}%")

for inputs, expected in dataset:
    pred = network.predict(inputs)[0]
    classe = 1 if pred >= 0.5 else 0

    print(f"x = {inputs}")
    print(f"Esperado: {expected[0]}")
    print(f"Previsto bruto: {pred}")
    print(f"Classe prevista: {classe}")
    print()