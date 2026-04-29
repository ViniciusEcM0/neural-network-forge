from activation import ACTIVATIONS
from neuron import Neuron

class Layer:
    """A layer of neurons, where each neuron receives the same inputs but has its own weights,
    bias, and activation function. The layer can be trained on a dataset to adjust the weights
    and biases of its neurons based on the error between predicted and true values."""
    
    def __init__(self,num_inputs, num_outputs, activation): # inputs = weights number, outputs = neurons number, activation = activation function
        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_outputs)] # Create a list of neurons for the layer, each with num_inputs and the specified activation function.
    
    def predict(self, inputs):
        predictions = [neuron.predict(inputs) for neuron in self.neurons] # Get the prediction from each neuron in the layer for the given inputs
        return predictions
    
    def forward(self, inputs):
        z_list = []
        outputs = []
        for neuron in self.neurons:
            a, z = neuron.forward(inputs)
            z_list.append(z)
            outputs.append(a)
        return outputs, z_list
    
    def backward(self, inputs, z_list, output_gradients, lr):
        if len(z_list) != len(self.neurons):
            raise ValueError("z_list precisa ter um z para cada neurônio da camada.")
        if len(output_gradients) != len(self.neurons):
            raise ValueError("output_gradients precisa ter um valor para cada neurônio da camada")

        input_grad_total = [0 for _ in inputs]
        for e, output_gradient in enumerate(output_gradients):
            input_gradients = self.neurons[e].backward(inputs, z_list[e], output_gradient, lr)
            for j in range(len(input_gradients)):
                input_grad_total[j] += input_gradients[j]
        return input_grad_total

    def train_example(self, inputs, y_true_list, lr): # Train the layer on a single example, where y_true_list is a list of true values for each neuron in the layer
        if len(y_true_list) != len(self.neurons):
            raise ValueError("O número de neurônios deve ser igual ao número de saídas em cada exemplo do dataset.")
        total_loss = 0
        for e, neuron in enumerate(self.neurons): # Train each neuron in the layer with the same inputs but different true values (y_true_list[e])
            total_loss += neuron.train_example(inputs, y_true_list[e], lr)
        return total_loss