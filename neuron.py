from activation import ACTIVATIONS
import random

class Neuron:
    """Single linear neuron with multiple inputs: y = dot(inputs, weights) + b."""
    def __init__(self, num_inputs, activation):
        self.weights = [random.random() for _ in range(num_inputs)]
        self.b = 0

        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        self.activation = ACTIVATIONS[activation]["function"]
        self.activation_derivative = ACTIVATIONS[activation]["derivative"]
    
    def compute_z(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Número de entradas diferente do número de pesos.")
        w_and_x = zip(self.weights, inputs)
        pred_sum = 0
        for w, x in w_and_x:
            pred_sum += x * w
        z = pred_sum + self.b
        return z
    
    def activate(self, z):
        return self.activation(z)

    def activation_derivative(self, z):
        return self.activation_derivative(z)

    def predict(self, inputs):
        z = self.compute_z(inputs)
        a = self.activate(z)
        return a
    
    def train_example(self, inputs, y_true, lr):
        z = self.compute_z(inputs)
        pred_y = self.activate(z)

        error = y_true - pred_y
        loss = error ** 2

        # Explicação matematica da derivação dos ajustes:
        # docs/derivacao_gradiente.md

        for i, w in enumerate(self.weights):
            gradient_w = -2 * error * self.activation_derivative(z) * inputs[i]
            self.weights[i] = self.weights[i] - lr * gradient_w

        gradient_b = -2 * error * self.activation_derivative(z)
        self.b = self.b - lr * gradient_b

        return loss
    
    def train(self, dataset, lr, epochs, log=True, log_interval=50):
        if len(dataset[0][0]) != len(self.weights):
            raise ValueError("O número de pesos deve ser igual ao número de entradas em cada exemplo do dataset.")

        loss_history = []
        for epoch in range(1, epochs+1):
            shuffled_dataset = dataset.copy()
            random.shuffle(shuffled_dataset)
            total_loss = 0

            for inputs, y_true in shuffled_dataset:
                loss = self.train_example(inputs, y_true, lr)
                total_loss += loss
            
            loss_history.append(total_loss / len(dataset))
            if log and epoch % log_interval == 0:
                print(f"Epoca: {epoch}  |  Perda(loss) total: {total_loss}  |  Perda(loss) média: {total_loss / len(dataset)}")
        return loss_history