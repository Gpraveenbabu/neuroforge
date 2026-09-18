from neuroforge.tensor import Parameter
from neuroforge.optim import SGD
from neuroforge.optim import Adam

def test_sgd_step():
    w = Parameter(1.0)

    optimizer = SGD([w], learning_rate=0.1)

    y = w ** 2
    y.backward()

    optimizer.step()

    assert w.data == 0.8


def test_sgd_zero_grad():
    w = Parameter(1.0)

    optimizer = SGD([w])

    w.grad = 5.0

    optimizer.zero_grad()

    assert w.grad == 0.0

def test_adam_updates_parameters():
    parameter = Parameter(1.0)
    parameter.grad = 0.5

    optimizer = Adam(
        [parameter],
        learning_rate=0.001
    )

    initial_value = parameter.data

    optimizer.step()

    assert parameter.data != initial_value


def test_adam_zero_grad():
    parameter = Parameter(1.0)
    parameter.grad = 0.5

    optimizer = Adam([parameter])

    optimizer.zero_grad()

    assert parameter.grad == 0.0