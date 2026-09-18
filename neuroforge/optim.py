class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0.0