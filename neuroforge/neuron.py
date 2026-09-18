import math
import random

from neuroforge.module import Module
from neuroforge.tensor import Parameter


class Neuron(Module):
    def __init__(self, input_size, activation="relu"):
        scale = math.sqrt(2.0 / input_size)

        self.weights = [
            Parameter(random.gauss(0.0, scale))
            for _ in range(input_size)
        ]

        self.bias = Parameter(0.0)
        self.activation = activation

    def __call__(self, inputs):
        output = self.bias

        for weight, input_value in zip(self.weights, inputs):
            output = output + weight * input_value

        if self.activation == "relu":
            return output.relu()

        if self.activation == "leaky_relu":
            return output.leaky_relu()

        raise ValueError(
            f"Unsupported activation: {self.activation}"
        )