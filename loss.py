import math

def mse(y_pred, y_true):
    error = y_pred - y_true
    return error ** 2

def mse_derivative(y_pred, y_true):
    error = y_pred - y_true
    return 2 * error


def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-12

    y_pred = max(min(y_pred, 1 - epsilon), epsilon)

    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))

def binary_cross_entropy_derivative(y_pred, y_true):
    epsilon = 1e-12

    y_pred = max(min(y_pred, 1 - epsilon), epsilon)

    return (y_pred - y_true) / (y_pred * (1 - y_pred))


LOSSES = {
    "mse":{
        "function": mse,
        "derivative": mse_derivative,
    },
    "binary_cross_entropy": {
        "function": binary_cross_entropy,
        "derivative": binary_cross_entropy_derivative,
    },
}