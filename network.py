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
    
    def forward(self, inputs):
        layer_outputs = [inputs]
        z_lists = []
        current_inputs = inputs

        for layer in self.layers:
            output, z = layer.forward(current_inputs)
            layer_outputs.append(output)
            z_lists.append(z)

            current_inputs = output

        return current_inputs, layer_outputs, z_lists

    def predict(self, inputs):
        outputs, _, _ = self.forward(inputs) # Get the final output of the network by passing the inputs through all layers
        return outputs
    
    def train_example(self, inputs, y_true_list, lr):
        if len(y_true_list) != len(self.layers[-1].neurons):
            raise ValueError("O número de saídas verdadeiras deve ser igual ao número de neurônios na camada de saída.")
        outputs, layer_outputs = self.forward(inputs) # Get the outputs of each layer for the given inputs
        loss = self.layers[-1].train_example(layer_outputs[-2], y_true_list, lr) # Train the output layer using the true values and the outputs from the last hidden layer as inputs
        return loss