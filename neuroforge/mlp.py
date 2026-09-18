from neuroforge.layer import Layer
from neuroforge.module import Module


class MLP(Module):
    def __init__(self, input_size, layer_sizes, activation="relu"):
        self.layers = []

        current_size = input_size

        for output_size in layer_sizes:
            self.layers.append(
                Layer(
                    current_size,
                    output_size,
                    activation=activation
                )
            )
            current_size = output_size

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs