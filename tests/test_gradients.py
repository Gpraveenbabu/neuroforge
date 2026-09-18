from neuroforge.tensor import Tensor
from neuroforge.neuron import Neuron
from neuroforge.layer import Layer
from neuroforge.mlp import MLP
from neuroforge.loss import MSELoss

def numerical_gradient(f, x, epsilon=1e-6):
    original = x.data

    x.data = original + epsilon
    positive = f().data

    x.data = original - epsilon
    negative = f().data

    x.data = original

    return (positive - negative) / (2 * epsilon)


def check_gradient(f, x):
    x.grad = 0.0

    y = f()
    y.backward()

    analytical = x.grad
    numerical = numerical_gradient(f, x)

    print(f"Analytical: {analytical}")
    print(f"Numerical:  {numerical}")

    assert abs(analytical - numerical) < 1e-5


def test_addition_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: x + 4.0,
        x
    )


def test_subtraction_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: x - 4.0,
        x
    )


def test_multiplication_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: x * 4.0,
        x
    )
def test_division_gradient():
    x = Tensor(6.0, requires_grad=True)

    check_gradient(
        lambda: x / 2.0,
        x
    )
def test_power_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: x ** 2,
        x
    )
def test_exp_gradient():
    x = Tensor(1.0, requires_grad=True)

    check_gradient(
        lambda: x.exp(),
        x
    )
def test_relu_positive_gradient():
    x = Tensor(2.0, requires_grad=True)

    check_gradient(
        lambda: x.relu(),
        x
    )
def test_relu_negative_gradient():
    x = Tensor(-2.0, requires_grad=True)

    check_gradient(
        lambda: x.relu(),
        x
    )
def test_leaky_relu_positive():
    x = Tensor(3.0, requires_grad=True)

    y = x.leaky_relu()

    y.backward()

    assert y.data == 3.0
    assert x.grad == 1.0


def test_leaky_relu_negative():
    x = Tensor(-3.0, requires_grad=True)

    y = x.leaky_relu()

    y.backward()

    assert y.data == -0.03
    assert x.grad == 0.01
def test_composed_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: (x ** 2 + 2.0 * x).relu(),
        x
    )
def test_reverse_addition_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: 2.0 + x,
        x
    )
def test_reverse_multiplication_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: 2.0 * x,
        x
    )
def test_reverse_subtraction_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: 2.0 - x,
        x
    )
def test_reverse_division_gradient():
    x = Tensor(3.0, requires_grad=True)

    check_gradient(
        lambda: 2.0 / x,
        x
    )
def test_neuron_gradient():
    neuron = Neuron(2)

    neuron.weights[0].data = 0.5
    neuron.weights[1].data = 0.5
    neuron.bias.data = 1.0

    x1 = Tensor(2.0, requires_grad=True)
    x2 = Tensor(3.0, requires_grad=True)

    y = neuron([x1, x2])

    y.backward()

    assert y.data == 3.5
    assert x1.grad == 0.5
    assert x2.grad == 0.5

    assert neuron.weights[0].grad == 2.0
    assert neuron.weights[1].grad == 3.0
    assert neuron.bias.grad == 1.0
def test_layer_gradient():
    layer = Layer(2, 2)

    layer.neurons[0].weights[0].data = 0.5
    layer.neurons[0].weights[1].data = 0.5
    layer.neurons[0].bias.data = 1.0

    layer.neurons[1].weights[0].data = 1.0
    layer.neurons[1].weights[1].data = -0.5
    layer.neurons[1].bias.data = 2.0

    x1 = Tensor(2.0, requires_grad=True)
    x2 = Tensor(1.0, requires_grad=True)

    outputs = layer([x1, x2])

    loss = outputs[0] + outputs[1]

    loss.backward()

    assert outputs[0].data == 2.5
    assert outputs[1].data == 3.5

    assert x1.grad == 1.5
    assert x2.grad == 0.0
def test_mlp_gradient():
    model = MLP(2, [2, 1])

    # First layer
    model.layers[0].neurons[0].weights[0].data = 0.5
    model.layers[0].neurons[0].weights[1].data = 0.5
    model.layers[0].neurons[0].bias.data = 1.0

    model.layers[0].neurons[1].weights[0].data = 1.0
    model.layers[0].neurons[1].weights[1].data = -0.5
    model.layers[0].neurons[1].bias.data = 2.0

    # Second layer
    model.layers[1].neurons[0].weights[0].data = 0.5
    model.layers[1].neurons[0].weights[1].data = 1.0
    model.layers[1].neurons[0].bias.data = 0.0

    x1 = Tensor(2.0, requires_grad=True)
    x2 = Tensor(1.0, requires_grad=True)

    output = model([x1, x2])[0]

    output.backward()

    assert output.data == 4.75

    assert x1.grad == 1.25
    assert x2.grad == -0.25

def test_mse_loss():
    prediction = Tensor(3.0, requires_grad=True)
    target = Tensor(2.0)

    loss_fn = MSELoss()
    loss = loss_fn(prediction, target)

    loss.backward()

    assert loss.data == 1.0
    assert prediction.grad == 2.0

def test_mean():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(4.0, requires_grad=True)

    mean = Tensor.mean([a, b])

    assert mean.data == 3.0


def test_mean_gradient():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(4.0, requires_grad=True)

    mean = Tensor.mean([a, b])

    mean.backward()

    assert a.grad == 0.5
    assert b.grad == 0.5
def test_mse_loss_batch():
    prediction_1 = Tensor(3.0, requires_grad=True)
    prediction_2 = Tensor(5.0, requires_grad=True)

    target_1 = Tensor(2.0)
    target_2 = Tensor(4.0)

    loss_fn = MSELoss()

    loss = loss_fn(
        [prediction_1, prediction_2],
        [target_1, target_2]
    )

    loss.backward()

    assert loss.data == 1.0
    assert prediction_1.grad == 1.0
    assert prediction_2.grad == 1.0