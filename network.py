from layer import Layer
from activation import ACTIVATIONS

class Network:
    """A class representing a feedforward neural network, which consists of multiple
    layers of neurons. The network can be trained using backpropagation to adjust the
    weights and biases of the neurons based on the error between predicted and true values."""

    def __init__(self, layers, internal_act, last_act): # layers = list of layer sizes, internal_act = activation function for hidden layers, last_act = activation function for output layer
        if internal_act not in ACTIVATIONS or last_act not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{internal_act}' ou '{last_act}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        if len(layers) < 2:
            raise ValueError("A rede deve ter pelo menos 2 camadas (entrada e saída).")
        
        self.layers = []
        # Create the layers of the network based on the specified layer sizes and activation functions
        for i in  range(len(layers)-1): 
            input_size = layers[i]
            output_size = layers[i+1]

            if i < len(layers)-2: # If it's not the last layer, use the internal activation function; otherwise, use the last activation function for the output layer
                activation = internal_act
            else:
                activation = last_act
            
            self.layers.append(Layer(input_size, output_size, activation)) # Create a new Layer with the specified input size, output size, and activation function, and add it to the network's layers
    
    def predict(self, inputs):
        for layer in self.layers: # Pass the inputs through each layer of the network, where the output of one layer becomes the input for the next layer
            inputs = layer.predict(inputs)
        return inputs