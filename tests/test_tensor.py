import pytest
import numpy as np
from neuroforge.tensor import Tensor
def test_tensor_list_initialization():
    tensor = Tensor([1, 2, 3])

    assert isinstance(tensor.data, np.ndarray)
    assert np.array_equal(tensor.data, np.array([1, 2, 3]))


def test_tensor_repr():
    tensor = Tensor(5)

    result = repr(tensor)

    assert "Tensor(data=5.0" in result
def test_sum_to_shape_singleton_dimension():
    gradient = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = Tensor._sum_to_shape(gradient, (1, 2))

    assert np.array_equal(result, np.array([[4.0, 6.0]]))


def test_mean_empty_list():
    with pytest.raises(ValueError, match="mean\\(\\) requires at least one tensor"):
        Tensor.mean([])