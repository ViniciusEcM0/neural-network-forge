class SGD:
    def update(self, value, gradient, lr, param_id=None):
        return value - lr * gradient
    

OPTIMIZERS = {
    "sgd": SGD()
}