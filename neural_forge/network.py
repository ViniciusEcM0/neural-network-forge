from .layer import Layer
from .activation import ACTIVATIONS
from .loss import LOSSES
from .optimizer import OPTIMIZERS
import random

class Network:
    """A class representing a feedforward neural network, which consists of multiple
    layers of neurons. The network can be trained using backpropagation to adjust the
    weights and biases of the neurons based on the error between predicted and true values."""

    def __init__(self, layers, internal_act, last_act, loss, init_method): # layers = list of layer sizes, internal_act = activation function for hidden layers, last_act = activation function for output layer
        if internal_act not in ACTIVATIONS or last_act not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{internal_act}' ou '{last_act}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        if len(layers) < 2:
            raise ValueError("A rede deve ter pelo menos 2 camadas (entrada e saída).")
        if loss not in LOSSES:
            raise ValueError(f"Função de perda '{loss}' não é suportada. Opções: {list(LOSSES.keys())}")
        
        self.loss_fn = LOSSES[loss]["function"]
        self.loss_derivative_fn = LOSSES[loss]["derivative"]
        
        self.layers = []
        # Create the layers of the network based on the specified layer sizes and activation functions
        for i in  range(len(layers)-1): 
            input_size = layers[i]
            output_size = layers[i+1]

            if i < len(layers)-2: # If it's not the last layer, use the internal activation function; otherwise, use the last activation function for the output layer
                activation = internal_act
            else:
                activation = last_act
            
            self.layers.append(Layer(input_size, output_size, activation,  init_method)) # Create a new Layer with the specified input size, output size, and activation function, and add it to the network's layers
    
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
    
    def backward(self, layer_outputs, z_lists, output_gradients):
        for layer_index in reversed(range(len(self.layers))):
            layer_input = layer_outputs[layer_index]
            layer_z_list = z_lists[layer_index]

            output_gradients = self.layers[layer_index].backward(layer_input, layer_z_list, output_gradients)

        return output_gradients
    
    def apply_gradients(self, lr, batch_size, optimizer):
        for layer in self.layers:
            layer.apply_gradients(lr, batch_size, optimizer)

    def zero_gradients(self):
        for layer in self.layers:
            layer.zero_gradients()

    def predict(self, inputs):
        outputs, _, _ = self.forward(inputs) # Get the final output of the network by passing the inputs through all layers
        return outputs
    
    def train_example(self, inputs, y_true_list):
        if len(y_true_list) != len(self.layers[-1].neurons):
            raise ValueError("O número de saídas verdadeiras deve ser igual ao número de neurônios na camada de saída.")
        
        outputs, layer_outputs, z_lists = self.forward(inputs) # Get the outputs of each layer for the given inputs
        total_loss = 0
        output_gradients = []
        for i in range(len(outputs)):
            y_pred = outputs[i]
            y_true = y_true_list[i]

            total_loss += self.loss_fn(y_pred, y_true)
            output_gradients.append(self.loss_derivative_fn(y_pred, y_true))
        self.backward(layer_outputs, z_lists, output_gradients)

        return total_loss

    def train(self, dataset, lr, epochs, batch_size, optimizer, log=True, log_interval=50):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Otimizador '{optimizer}' não é suportado. Opções: {list(OPTIMIZERS.keys())}")
        
        optimizer_fn = OPTIMIZERS[optimizer]

        loss_history = []

        for epoch in range(1, epochs+1):
            shuffled_dataset = dataset.copy()
            random.shuffle(shuffled_dataset)

            total_loss = 0
            
            for start in range(0, len(shuffled_dataset), batch_size):
                batch = shuffled_dataset[start:start+batch_size]

                self.zero_gradients()

                for inputs, y_true_list in batch:
                    loss = self.train_example(inputs, y_true_list)
                    total_loss += loss
                
                self.apply_gradients(lr, len(batch), optimizer_fn)

            avg_loss = total_loss/(len(dataset) * len(self.layers[-1].neurons))
            loss_history.append(avg_loss)

            if log and epoch % log_interval == 0:
                print(f"Época {epoch}  |  total_loss: {total_loss}  |  loss: {avg_loss}")

        return loss_history
    
    def evaluate(self, dataset):
        correct = 0

        for inputs, y_true_list in dataset:
            pred = self.predict(inputs)[0]
            pred_class = 1 if pred >= 0.5 else 0
            true_class = y_true_list[0]

            if pred_class == true_class:
                correct += 1

        accuracy = correct / len(dataset)
        return accuracy
