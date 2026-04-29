import math

def linear(z):
    return z

def linear_derivative(z):
    return 1


def relu(z):
    return max(0, z)

def relu_derivative(z):
    return 1 if z > 0 else 0


def leaky_relu(z):
    if z > 0:
        return z
    return 0.01 * z

def leaky_relu_derivative(z):
    if z > 0:
        return 1
    return 0.01


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


ACTIVATIONS = {
    "linear":{
        "function": linear,
        "derivative": linear_derivative,
    },
    "relu":{
        "function": relu,
        "derivative": relu_derivative,
    },
    "leaky_relu":{
        "function": leaky_relu,
        "derivative": leaky_relu_derivative,
    },
    "sigmoid":{
        "function": sigmoid,
        "derivative": sigmoid_derivative,
    },
}