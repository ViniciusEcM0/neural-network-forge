from layer import Layer
from network import Network
from data import dataset

lr = 0.01
epochs = 500

network = Network(layers=[2, 4, 3, 1], internal_act="relu", last_act="relu")
loss_history = network.train_example([1, 2], [10], 0.01)


