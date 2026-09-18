import random
import matplotlib.pyplot as plt
from neuroforge.mlp import MLP
from neuroforge.optim import SGD
from neuroforge.tensor import Tensor
from neuroforge.training import train
from neuroforge.loss import MSELoss
from pathlib import Path

random.seed(42)

model = MLP(
    input_size=2,
    layer_sizes=[4, 1],
    activation="leaky_relu"
)

optimizer = SGD(
    model.parameters(),
    learning_rate=0.05
)

dataset = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

dataset = [
    ([Tensor(x) for x in inputs], Tensor(target))
    for inputs, target in dataset
]

history = train(
    model,
    dataset,
    optimizer,
    epochs=1000,
    loss_fn=MSELoss(),
    batch_size=4,
    verbose=False
)

print(f"Final loss: {history[-1]:.6f}")

for inputs, target in dataset:
    prediction = model(inputs)[0]

    print(
        f"Input: {[x.data for x in inputs]} "
        f"Target: {target.data:.0f} "
        f"Prediction: {prediction.data:.4f}"
    )
Path("results").mkdir(exist_ok=True)

plt.plot(history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("NeuroForge XOR Training Loss")
plt.savefig("results/xor_loss.png")
plt.show()
