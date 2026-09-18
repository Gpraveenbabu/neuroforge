from neuroforge.checkpoint import (
    save_checkpoint,
    load_checkpoint
)
from neuroforge.mlp import MLP


def test_save_and_load_checkpoint(tmp_path):
    model = MLP(1, [2, 1])

    original_values = [
        parameter.data
        for parameter in model.parameters()
    ]

    checkpoint_path = tmp_path / "model.pkl"

    save_checkpoint(model, checkpoint_path)

    for parameter in model.parameters():
        parameter.data = 999.0

    load_checkpoint(model, checkpoint_path)

    restored_values = [
        parameter.data
        for parameter in model.parameters()
    ]

    assert restored_values == original_values
