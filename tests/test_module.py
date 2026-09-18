from neuroforge.module import Module
from neuroforge.tensor import Parameter


def test_nested_module_parameters():
    class Child(Module):
        def __init__(self):
            self.weight = Parameter(1.0)

    class Parent(Module):
        def __init__(self):
            self.child = Child()

    model = Parent()
    params = model.parameters()

    assert len(params) == 1
    assert params[0].data == 1.0
