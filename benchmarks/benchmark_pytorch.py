import time
import random
import torch
import torch.nn as nn

from neuroforge.mlp import MLP
from neuroforge.optim import SGD
from neuroforge.tensor import Tensor
from neuroforge.training import train
from neuroforge.loss import MSELoss


# Dataset
data = [
    (1.0, 2.0),
    (2.0, 4.0),
    (3.0, 6.0),
    (4.0, 8.0),
    (5.0, 10.0),
]

epochs = 100


# -------------------------
# NeuroForge benchmark
# -------------------------

random.seed(42)

model = MLP(1, [1], activation="leaky_relu")

optimizer = SGD(
    model.parameters(),
    learning_rate=0.01
)

dataset = [
    (Tensor(x), Tensor(y))
    for x, y in data
]

start = time.perf_counter()

history = train(
    model,
    dataset,
    optimizer,
    epochs=epochs,
    loss_fn=MSELoss(),
    batch_size=5
)

neuroforge_time = time.perf_counter() - start

print("NeuroForge")
print(f"Initial loss: {history[0]:.6f}")
print(f"Final loss: {history[-1]:.6f}")
print(f"Training time: {neuroforge_time:.6f} seconds")


# -------------------------
# PyTorch benchmark
# -------------------------

torch.manual_seed(42)

torch_model = nn.Sequential(
    nn.Linear(1, 1),
    nn.LeakyReLU()
)

torch_optimizer = torch.optim.SGD(
    torch_model.parameters(),
    lr=0.01
)

loss_fn = nn.MSELoss()

x_train = torch.tensor(
    [[x] for x, _ in data],
    dtype=torch.float32
)

y_train = torch.tensor(
    [[y] for _, y in data],
    dtype=torch.float32
)

start = time.perf_counter()

torch_model.train()

with torch.no_grad():
    initial_loss = loss_fn(
        torch_model(x_train),
        y_train
    ).item()

for _ in range(epochs):
    torch_optimizer.zero_grad()

    predictions = torch_model(x_train)
    loss = loss_fn(predictions, y_train)

    loss.backward()
    torch_optimizer.step()

with torch.no_grad():
    final_loss = loss_fn(
        torch_model(x_train),
        y_train
    ).item()

pytorch_time = time.perf_counter() - start

print("\nPyTorch")
print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss: {final_loss:.6f}")
print(f"Training time: {pytorch_time:.6f} seconds")


# -------------------------
# Comparison
# -------------------------

print("\nComparison")
print(f"NeuroForge: {neuroforge_time:.6f} seconds")
print(f"PyTorch:    {pytorch_time:.6f} seconds")