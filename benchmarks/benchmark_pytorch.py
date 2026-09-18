import time
import random
import statistics
import numpy as np
import torch
import torch.nn as nn

from neuroforge.mlp import MLP
from neuroforge.optim import SGD
from neuroforge.tensor import Tensor
from neuroforge.training import train
from neuroforge.loss import MSELoss
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA = [
    (x / 10.0, 2.0 * (x / 10.0))
    for x in range(1, 101)
]

EPOCHS = 50
REPETITIONS = 10


def benchmark_neuroforge():
    times = []
    final_losses = []

    for _ in range(REPETITIONS):
        random.seed(42)

        model = MLP(1, [8, 1], activation="leaky_relu")
        optimizer = SGD(
            model.parameters(),
            learning_rate=0.001
        )

        dataset = [
            (Tensor(x), Tensor(y))
            for x, y in DATA
        ]

        start = time.perf_counter()

        history = train(
            model,
            dataset,
            optimizer,
            epochs=EPOCHS,
            loss_fn=MSELoss(),
            batch_size=len(dataset),
            verbose=False
        )

        elapsed = time.perf_counter() - start

        times.append(elapsed)
        final_losses.append(history[-1])

    return times, final_losses


def benchmark_pytorch():
    times = []
    final_losses = []

    for _ in range(REPETITIONS):
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(1, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
            nn.LeakyReLU()
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.001
        )

        loss_fn = nn.MSELoss()

        x_train = torch.tensor(
            [[x] for x, _ in DATA],
            dtype=torch.float32
        )

        y_train = torch.tensor(
            [[y] for _, y in DATA],
            dtype=torch.float32
        )

        start = time.perf_counter()

        for _ in range(EPOCHS):
            optimizer.zero_grad()

            predictions = model(x_train)
            loss = loss_fn(predictions, y_train)

            loss.backward()
            optimizer.step()

        elapsed = time.perf_counter() - start

        times.append(elapsed)
        final_losses.append(loss.item())

    return times, final_losses


neuroforge_times, neuroforge_losses = benchmark_neuroforge()
pytorch_times, pytorch_losses = benchmark_pytorch()

print("\nBenchmark Results")
print("=================")

print(
    f"NeuroForge average: "
    f"{statistics.mean(neuroforge_times):.6f} seconds"
)

print(
    f"PyTorch average:    "
    f"{statistics.mean(pytorch_times):.6f} seconds"
)
neuroforge_avg = statistics.mean(neuroforge_times)
pytorch_avg = statistics.mean(pytorch_times)

speedup = neuroforge_avg / pytorch_avg

print(f"NeuroForge is {speedup:.1f}x slower than PyTorch")
print(
    f"\nNeuroForge final loss: "
    f"{statistics.mean(neuroforge_losses):.6f}"
)

print(
    f"PyTorch final loss:    "
    f"{statistics.mean(pytorch_losses):.6f}"
)