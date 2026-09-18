import numpy as np

from neuroforge.tensor import Tensor


def test_tensor_stores_numpy_array():
    tensor = Tensor(np.array([1.0, 2.0, 3.0]))

    assert isinstance(tensor.data, np.ndarray)
    assert tensor.data.shape == (3,)
    assert np.array_equal(
        tensor.data,
        np.array([1.0, 2.0, 3.0])
    )
def test_numpy_tensor_addition():
    tensor_a = Tensor(np.array([1.0, 2.0, 3.0]))
    tensor_b = Tensor(np.array([4.0, 5.0, 6.0]))

    result = tensor_a + tensor_b

    assert np.array_equal(
        result.data,
        np.array([5.0, 7.0, 9.0])
    )

def test_numpy_tensor_multiplication():
    tensor_a = Tensor(np.array([1.0, 2.0, 3.0]))
    tensor_b = Tensor(np.array([4.0, 5.0, 6.0]))

    result = tensor_a * tensor_b

    assert np.array_equal(
        result.data,
        np.array([4.0, 10.0, 18.0])
    )

def test_numpy_tensor_matrix_multiplication():
    tensor_a = Tensor(
        np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
    )

    tensor_b = Tensor(
        np.array([
            [5.0, 6.0],
            [7.0, 8.0]
        ])
    )

    result = tensor_a @ tensor_b

    expected = np.array([
        [19.0, 22.0],
        [43.0, 50.0]
    ])

    assert np.array_equal(result.data, expected)
def test_matrix_multiplication_gradients():
    tensor_a = Tensor(
        np.array([[1.0, 2.0]]),
        requires_grad=True
    )

    tensor_b = Tensor(
        np.array([[3.0], [4.0]]),
        requires_grad=True
    )

    result = tensor_a @ tensor_b

    result.backward()

    assert np.array_equal(
        tensor_a.grad,
        np.array([[3.0, 4.0]])
    )

    assert np.array_equal(
        tensor_b.grad,
        np.array([[1.0], [2.0]])
    )
def test_numpy_tensor_broadcasting_addition():
    tensor_a = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        requires_grad=True
    )

    tensor_b = Tensor(
        np.array([10.0, 20.0]),
        requires_grad=True
    )

    result = tensor_a + tensor_b

    expected = np.array([
        [11.0, 22.0],
        [13.0, 24.0]
    ])

    np.testing.assert_array_equal(result.data, expected)
def test_numpy_tensor_broadcasting_gradients():
    tensor_a = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        requires_grad=True
    )

    tensor_b = Tensor(
        np.array([10.0, 20.0]),
        requires_grad=True
    )

    result = tensor_a + tensor_b

    result.backward()

    np.testing.assert_array_equal(
        tensor_a.grad,
        np.ones((2, 2))
    )

    np.testing.assert_array_equal(
        tensor_b.grad,
        np.array([2.0, 2.0])
    )
def test_numpy_tensor_broadcasting_multiplication():
    tensor_a = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        requires_grad=True
    )

    tensor_b = Tensor(
        np.array([10.0, 20.0]),
        requires_grad=True
    )

    result = tensor_a * tensor_b

    expected = np.array([
        [10.0, 40.0],
        [30.0, 80.0]
    ])

    np.testing.assert_array_equal(
        result.data,
        expected
    )
def test_numpy_tensor_broadcasting_multiplication_gradients():
    tensor_a = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        requires_grad=True
    )

    tensor_b = Tensor(
        np.array([10.0, 20.0]),
        requires_grad=True
    )

    result = tensor_a * tensor_b

    result.backward()

    np.testing.assert_array_equal(
        tensor_a.grad,
        np.array([[10.0, 20.0], [10.0, 20.0]])
    )

    np.testing.assert_array_equal(
        tensor_b.grad,
        np.array([4.0, 6.0])
    )