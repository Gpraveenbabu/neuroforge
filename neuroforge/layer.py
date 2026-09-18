from neuroforge.module import Module
from neuroforge.neuron import Neuron


class Layer(Module):
    def __init__(self, input_size, output_size, activation="relu"):
        self.neurons = [
            Neuron(input_size, activation=activation)
            for _ in range(output_size)
        ]

    def __call__(self, inputs):
        return [
            neuron(inputs)
            for neuron in self.neurons
        ]