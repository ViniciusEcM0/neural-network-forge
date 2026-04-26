from activation import ACTIVATIONS
from neuron import Neuron
import random

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

    def train_example(self, inputs, y_true_list, lr): # Train the layer on a single example, where y_true_list is a list of true values for each neuron in the layer
        if len(y_true_list) != len(self.neurons):
            raise ValueError("O número de neurônios deve ser igual ao número de saídas em cada exemplo do dataset.")
        total_loss = 0
        for e, neuron in enumerate(self.neurons): # Train each neuron in the layer with the same inputs but different true values (y_true_list[e])
            total_loss += neuron.train_example(inputs, y_true_list[e], lr)
        return total_loss
    
    # Train the layer on the entire dataset for a specified number of epochs, and return the history of losses.
    def train(self, dataset, lr, epochs, log=True, log_interval=50):
        if len(dataset[0][1]) != len(self.neurons):
            raise ValueError("O número de neurônios deve ser igual ao número de saídas em cada exemplo do dataset.")
        loss_history = []
        for epoch in range(1, epochs+1): # Training loop for each epoch
            shuffled_dataset = dataset.copy()
            random.shuffle(shuffled_dataset)
            total_loss = 0
            for inputs, y_true_list in shuffled_dataset: # Training loop for each example in the dataset
                loss = self.train_example(inputs, y_true_list, lr) # Train the layer on the example and get the loss
                total_loss += loss
            loss_history.append(total_loss / len(dataset))
            if log and epoch % log_interval == 0:
                print(f"Epoca: {epoch}  |  Perda(loss) total: {total_loss}  |  Perda(loss) média: {total_loss / (len(dataset) * len(self.neurons))}")
        return loss_history
