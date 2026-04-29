from layer import Layer
from network import Network
from data import generate_circle_dataset


train_dataset = generate_circle_dataset(1000, 0.6)
test_dataset = generate_circle_dataset(200, 0.6)

lr = 0.001
epochs = 500

network = Network(
    layers=[2, 8, 8, 1],
    internal_act="relu",
    last_act="sigmoid"
)
network.train(train_dataset, lr, epochs)

testes = [
    ([0, 0], [1]),
    ([0.2, 0.2], [1]),
    ([0.5, 0], [1]),
    ([0.7, 0], [0]),
    ([1, 1], [0]),
    ([-0.3, 0.4], [1]),
    ([-0.8, 0.2], [0]),
]

for inputs, expected in testes:
    pred = network.predict(inputs)[0]
    classe = 1 if pred >= 0.5 else 0

    print(f"x = {inputs}")
    print(f"Esperado: {expected[0]}")
    print(f"Previsto bruto: {pred}")
    print(f"Classe prevista: {classe}")
    print()