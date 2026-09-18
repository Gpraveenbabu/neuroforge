from neuroforge.tensor import Parameter
from neuroforge.optim import SGD


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