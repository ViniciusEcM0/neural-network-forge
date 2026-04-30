from .activation import ACTIVATIONS
from .initialization import INITIALIZERS
from .optimizer import SGD

class Neuron:
    """Single linear neuron with multiple inputs: y = dot(inputs, weights) + b."""
    
    def __init__(self, num_inputs, activation, init_method): # num_inputs = weights number, activation = activation function
        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        if init_method not in INITIALIZERS:
            raise ValueError(f"Método de inicialização '{init_method}' não é suportado. Opções: {list(INITIALIZERS.keys())}")

        self.weights = INITIALIZERS[init_method](num_inputs) # Initialize the weights of the neuron using the specified initialization method, which generates a list of weights based on the number of inputs and outputs
        self.b = 0

        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        self.activation = ACTIVATIONS[activation]["function"]
        self.activation_derivative_fn = ACTIVATIONS[activation]["derivative"]

        self.weight_gradients_sum = [0 for _ in range(num_inputs)]
        self.bias_gradients_sum = 0
    
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
    
    def backward(self, inputs, z, output_gradient):
        activation_gradient = self.derivative(z)
        raw_gradient = output_gradient * activation_gradient

        old_w = self.weights.copy()

        for i in range(len(self.weights)):
            weight_gradient = raw_gradient * inputs[i]
            self.weight_gradients_sum[i] += weight_gradient
        
        self.bias_gradients_sum += raw_gradient

        return [raw_gradient * w for w in old_w]
    
    def apply_gradients(self, lr, batch_size, optimizer):
        for i in range(len(self.weights)):
            avg_gradient = self.weight_gradients_sum[i] / batch_size
            self.weights[i] = optimizer.update(self.weights[i], avg_gradient, lr, param_id=f"{id(self)}.weight.{i}")
        avg_bias_gradient = self.bias_gradients_sum / batch_size
        self.b = optimizer.update(self.b, avg_bias_gradient, lr, param_id=f"{id(self)}.bias")
    
    def zero_gradients(self):
        self.weight_gradients_sum = [0 for _ in self.weights]
        self.bias_gradients_sum = 0
    
    def train_example(self, inputs, y_true):
        pred_y, z = self.forward(inputs)

        error = pred_y - y_true

        loss = error ** 2

        # Matematic derivation of the gradients:
        # docs/derivacao_gradiente.md

        output_gradient = 2 * error

        self.backward(inputs, z, output_gradient)

        return loss