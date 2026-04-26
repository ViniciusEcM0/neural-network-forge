from activation import ACTIVATIONS
from neuron import Neuron
import random

class Layer:
    def __init__(self,num_inputs, num_outputs, activation): # inputs = quantidade de pesos p/neuronio, outputs = quantidade de neurônios
        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_outputs)]
    
    def predict(self, inputs):
        predictions = [neuron.predict(inputs) for neuron in self.neurons]
        return predictions

    def train_example(self, inputs, y_true_list, lr):
        if len(y_true_list) != len(self.neurons):
            raise ValueError("O número de neurônios deve ser igual ao número de saídas em cada exemplo do dataset.")
        total_loss = 0
        for e, neuron in enumerate(self.neurons):
            total_loss += neuron.train_example(inputs, y_true_list[e], lr)
        return total_loss
    
    def train(self, dataset, lr, epochs, log=True, log_interval=50):
        if len(dataset[0][1]) != len(self.neurons):
            raise ValueError("O número de neurônios deve ser igual ao número de saídas em cada exemplo do dataset.")
        loss_history = []
        for epoch in range(1, epochs+1):
            shuffled_dataset = dataset.copy()
            random.shuffle(shuffled_dataset)
            total_loss = 0
            for inputs, y_true_list in shuffled_dataset:
                loss = self.train_example(inputs, y_true_list, lr)
                total_loss += loss
            loss_history.append(total_loss / len(dataset))
            if log and epoch % log_interval == 0:
                print(f"Epoca: {epoch}  |  Perda(loss) total: {total_loss}  |  Perda(loss) média: {total_loss / (len(dataset) * len(self.neurons))}")
        return loss_history
