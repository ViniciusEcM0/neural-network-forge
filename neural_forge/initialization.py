import random
import math

def small_random(num_inputs):
    return [random.uniform(-0.1, 0.1) for _ in range(num_inputs)]

def xavier(num_inputs):
    limit = math.sqrt(1 / num_inputs)
    return [random.uniform(-limit, limit) for _ in range(num_inputs)]

def he(num_inputs):
    limit = math.sqrt(2 / num_inputs)
    return [random.uniform(-limit, limit) for _ in range(num_inputs)]


INITIALIZERS = {
    "small_random": small_random,
    "xavier": xavier,
    "he": he
}