from neural_forge.network import Network
from data import generate_circle_dataset


train_dataset = generate_circle_dataset(1000, 0.6)
test_dataset = generate_circle_dataset(200, 0.6)

lr = 0.003
epochs = 1500
batch_size = 32

network = Network(
    layers=[2, 4, 1],
    internal_act="relu",
    last_act="sigmoid",
    loss="binary_cross_entropy",
    init_method="he"
)
network.train(train_dataset, lr, epochs, batch_size, "adam")

train_accuracy = network.evaluate(train_dataset)
test_accuracy = network.evaluate(test_dataset)

print(f"Acurácia treinamento: {train_accuracy * 100:.2f}%")
print(f"Acurácia teste: {test_accuracy * 100:.2f}%")
