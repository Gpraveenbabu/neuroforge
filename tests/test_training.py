from neuroforge.loss import MSELoss
from neuroforge.mlp import MLP
from neuroforge.optim import SGD
from neuroforge.tensor import Tensor
from neuroforge.training import train


def test_batch_training_improves_prediction():
    model = MLP(1, [1], activation="leaky_relu")

    neuron = model.layers[0].neurons[0]
    neuron.weights[0].data = 1.0
    neuron.bias.data = 0.0

    optimizer = SGD(
        model.parameters(),
        learning_rate=0.01
    )

    dataset = [
        (Tensor(1.0), Tensor(2.0)),
        (Tensor(2.0), Tensor(4.0)),
    ]

    prediction_before = model([Tensor(1.0)])[0].data

    history = train(
    model,
    dataset,
    optimizer,
    epochs=20,
    loss_fn=MSELoss(),
    batch_size=2
    )

    assert len(history) == 20
    assert history[-1] < history[0]

    prediction_after = model([Tensor(1.0)])[0].data

    assert abs(prediction_after - 2.0) < abs(
        prediction_before - 2.0
    )
def test_training_default_loss_single_batch():
    model = MLP(1, [1], activation="leaky_relu")

    neuron = model.layers[0].neurons[0]
    neuron.weights[0].data = 1.0
    neuron.bias.data = 0.0

    optimizer = SGD(
        model.parameters(),
        learning_rate=0.01
    )

    dataset = [
        (Tensor(1.0), Tensor(2.0)),
    ]

    history = train(
        model,
        dataset,
        optimizer,
        epochs=2
    )

    assert len(history) == 2
def test_vector_input_training():
    model = MLP(
        2,
        [4, 1],
        activation="leaky_relu"
    )

    optimizer = SGD(
        model.parameters(),
        learning_rate=0.05
    )

    dataset = [
        ([Tensor(0.0), Tensor(0.0)], Tensor(0.0)),
        ([Tensor(0.0), Tensor(1.0)], Tensor(1.0)),
        ([Tensor(1.0), Tensor(0.0)], Tensor(1.0)),
        ([Tensor(1.0), Tensor(1.0)], Tensor(0.0)),
    ]

    history = train(
        model,
        dataset,
        optimizer,
        epochs=10,
        loss_fn=MSELoss(),
        batch_size=4,
        verbose=False
    )

    assert len(history) == 10
    assert all(loss >= 0 for loss in history)