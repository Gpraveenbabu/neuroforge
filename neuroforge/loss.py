from neuroforge.module import Module
from neuroforge.tensor import Tensor


class MSELoss(Module):
    def __call__(self, prediction, target):
        if isinstance(prediction, list):
            losses = [
                (pred - expected) ** 2
                for pred, expected in zip(prediction, target)
            ]

            return Tensor.mean(losses)

        return (prediction - target) ** 2