from layer import Layer
from network import Network
from data import dataset

lr = 0.01
epochs = 500

layer = Layer(2, 3, "leaky_relu") # 2 entradas, 3 neurônios, função de ativação ReLU
loss_history = layer.train(dataset, lr, epochs)
print("\nPesos finais:")
for i, neuron in enumerate(layer.neurons):
    print(f"Neurônio {i+1}: Pesos: {neuron.weights}  |  Bias: {neuron.b}")
