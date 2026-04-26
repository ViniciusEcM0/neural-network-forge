from layer import Layer
from activation import ACTIVATIONS

class Network:
    def __init__(self, layers, internal_act, last_act):
        if internal_act not in ACTIVATIONS or last_act not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{internal_act}' ou '{last_act}' não é suportada. Opções: {list(ACTIVATIONS.keys())}")
        if len(layers) < 2:
            raise ValueError("A rede deve ter pelo menos 2 camadas (entrada e saída).")
        self.layers = []
        for i in  range(len(layers)-1):
            input_size = layers[i]
            output_size = layers[i+1]

            if i < len(layers)-2:
                activation = internal_act
            else:
                activation = last_act
            
            self.layers.append(Layer(input_size, output_size, activation))
    
    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.predict(inputs)
        return inputs