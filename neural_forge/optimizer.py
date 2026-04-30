import math

class SGD:
    def update(self, value, gradient, lr, param_id=None):
        return value - lr * gradient

class Momentum:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = {}

    def update(self, value, gradient, lr, param_id=None):
        if param_id == None:
            raise ValueError("Momentum precisa de param_id para guardar a velocidade de cada parâmetro.")
        
        previous_velocity = self.velocity.get(param_id, 0)

        velocity = self.momentum * previous_velocity - lr * gradient
        self.velocity[param_id] = velocity

        return value + velocity

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}
    
    def update(self, value, gradient, lr, param_id=None):
        if param_id == None:
            raise ValueError("Adam precisa de param_id para guardar os momentos de cada parâmetro.")

        previous_m = self.m.get(param_id, 0)
        previous_v = self.v.get(param_id, 0)
        previous_t = self.t.get(param_id, 0)

        current_t = previous_t + 1

        current_m = self.beta1 * previous_m + (1 - self.beta1) * gradient
        current_v = self.beta2 * previous_v + (1 - self.beta2) * (gradient ** 2)

        corrected_m = current_m / (1 - self.beta1 ** current_t)
        corrected_v = current_v / (1 - self.beta2 ** current_t)

        new_value = value - lr * corrected_m / (math.sqrt(corrected_v) + self.epsilon)

        self.m[param_id] = current_m
        self.v[param_id] = current_v
        self.t[param_id] = current_t

        return new_value

    

OPTIMIZERS = {
    "sgd": SGD(),
    "momentum": Momentum(momentum=0.9),
    "adam": Adam()
}