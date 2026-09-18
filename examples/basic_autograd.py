import random
import matplotlib.pyplot as plt
random.seed(42)
from neuroforge.mlp import MLP
from neuroforge.optim import SGD
from neuroforge.tensor import Tensor
from neuroforge.training import train
from neuroforge.loss import MSELoss
model = MLP(1, [1] , activation="leaky_relu")
optimizer = SGD(
    model.parameters(),
    learning_rate=0.01
)

dataset = [
    (Tensor(1.0), Tensor(2.0)),
    (Tensor(2.0), Tensor(4.0)),
    (Tensor(3.0), Tensor(6.0)),
    (Tensor(4.0), Tensor(8.0)),
    (Tensor(5.0), Tensor(10.0))
]

history = train(
    model,
    dataset,
    optimizer,
    epochs=100,
    loss_fn=MSELoss(),
    batch_size=5
)

print()
print(f"Initial loss: {history[0]:.6f}")
print(f"Final loss: {history[-1]:.6f}")
print()
for value in [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]:
    x = Tensor(value)
    prediction = model([x])[0]

    print(
        f"x={value:.1f}, "
        f"prediction={prediction.data:.4f}, "
        f"expected={2.0 * value:.4f}"
    )
plt.plot(history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("NeuroForge Training Loss")
plt.grid(True)
plt.savefig(
    "results/training_loss.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()