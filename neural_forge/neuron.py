from neural_forge.activation import ACTIVATIONS
import random

class Neuron:
    """Single linear neuron with multiple inputs: y = dot(inputs, weights) + b."""
    
    def __init__(self, num_inputs, activation): # num_inputs = weights number, activation = activation function
        self.weights = [random.random() for _ in range(num_inputs)] # initial random weights between 0 and 1
        self.b = 0

        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        self.activation = ACTIVATIONS[activation]["function"]
        self.activation_derivative_fn = ACTIVATIONS[activation]["derivative"]
    
    def compute_z(self, inputs): # z = dot(inputs, weights) + b
        if len(inputs) != len(self.weights):
            raise ValueError("Número de entradas diferente do número de pesos.")
        w_and_x = zip(self.weights, inputs) # [(w1, x1), (w2, x2), ...]
        pred_sum = 0 
        for w, x in w_and_x: # pred_sum = w1*x1 + w2*x2 + ...
            pred_sum += x * w
        z = pred_sum + self.b
        return z
    
    def activate(self, z):
        return self.activation(z)

    def derivative(self, z):
        return self.activation_derivative_fn(z)

    def predict(self, inputs):
        z = self.compute_z(inputs)
        a = self.activate(z)
        return a
    
    def forward(self, inputs):
        z = self.compute_z(inputs)
        a = self.activate(z)
        return a, z
    
    def backward(self, inputs, z, output_gradient, lr):
        activation_gradient = self.derivative(z)
        raw_gradient = output_gradient * activation_gradient
        gradient_w = [raw_gradient * inputs[i] for i in range(len(self.weights))]
        gradient_b = raw_gradient
        old_w = self.weights.copy()

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - gradient_w[i] * lr
        self.b = self.b - gradient_b * lr

        return [raw_gradient * w for w in old_w]
    
    def train_example(self, inputs, y_true, lr):
        pred_y, z = self.forward(inputs)

        error = pred_y - y_true

        loss = error ** 2

        # Matematic derivation of the gradients:
        # docs/derivacao_gradiente.md

        output_gradient = 2 * error

        self.backward(inputs, z, output_gradient, lr)

        return loss