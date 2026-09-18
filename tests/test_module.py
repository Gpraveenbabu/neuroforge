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
def test_state_dict():
    from neuroforge.mlp import MLP

    model = MLP(2, [3, 1])

    state = model.state_dict()

    assert len(state) == len(model.parameters())


def test_load_state_dict():
    from neuroforge.mlp import MLP

    model = MLP(2, [3, 1])

    state = model.state_dict()

    for parameter in model.parameters():
        parameter.data = 999.0

    model.load_state_dict(state)

    restored_state = model.state_dict()

    assert restored_state == state