import numpy as np
class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=float)

        self.data = data if isinstance(data, np.ndarray) else float(data)
        self.requires_grad = requires_grad
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set()
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    @staticmethod
    def _sum_to_shape(gradient, shape):
        if not isinstance(gradient, np.ndarray):
            return gradient

        while gradient.ndim > len(shape):
            gradient = gradient.sum(axis=0)

        for axis, size in enumerate(shape):
            if size == 1:
                gradient = gradient.sum(
                    axis=axis,
                    keepdims=True
                )

        return gradient
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                gradient = Tensor._sum_to_shape(
                    out.grad,
                    self.data.shape if isinstance(self.data, np.ndarray) else ()
                )
                self.grad += gradient

            if other.requires_grad:
                gradient = Tensor._sum_to_shape(
                    out.grad,
                    other.data.shape if isinstance(other.data, np.ndarray) else ()
                )
                other.grad += gradient

        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                gradient = out.grad * other.data

                if isinstance(self.data, np.ndarray):
                    gradient = Tensor._sum_to_shape(
                        gradient,
                        self.data.shape
                    )

                self.grad += gradient

            if other.requires_grad:
                gradient = out.grad * self.data

                if isinstance(other.data, np.ndarray):
                    gradient = Tensor._sum_to_shape(
                        gradient,
                        other.data.shape
                    )

                other.grad += gradient

        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return Tensor(other) - self

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += out.grad

            if other.requires_grad:
                other.grad -= out.grad

        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / other.data) * out.grad

            if other.requires_grad:
                self.grad += 0.0
                other.grad += (-self.data / (other.data ** 2)) * out.grad

        out._backward = _backward

        return out
    
    def __pow__(self, exponent):
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad
        )

        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad += (
                    exponent
                    * (self.data ** (exponent - 1))
                    * out.grad
                )

        out._backward = _backward

        return out
    
    def exp(self):
        import math

        out = Tensor(
            math.exp(self.data),
            requires_grad=self.requires_grad
        )

        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(
            max(0.0, self.data),
            requires_grad=self.requires_grad
        )

        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward

        return out
    
    def leaky_relu(self, negative_slope=0.01):
        out = Tensor(
            self.data if self.data > 0 else negative_slope * self.data,
            requires_grad=self.requires_grad
        )

        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad += (
                    1.0 if self.data > 0 else negative_slope
                ) * out.grad

        out._backward = _backward

        return out
    @staticmethod
    def mean(tensors):
        if not tensors:
            raise ValueError("mean() requires at least one tensor")

        total = tensors[0]

        for tensor in tensors[1:]:
            total = total + tensor

        return total / len(tensors)
    def backward(self):
        topo = []
        visited = set()

        def build_topology(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build_topology(parent)

                topo.append(node)

        build_topology(self)

        self.grad = (
            np.ones_like(self.data)
            if isinstance(self.data, np.ndarray)
            else 1.0
        )

        for node in reversed(topo):
            node._backward()

class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(
            data,
            requires_grad=requires_grad
        )