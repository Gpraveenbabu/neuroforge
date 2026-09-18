import pickle
from pathlib import Path


def save_checkpoint(model, path):
    path = Path(path)

    parameters = [
        parameter.data
        for parameter in model.parameters()
    ]

    with path.open("wb") as file:
        pickle.dump(parameters, file)


def load_checkpoint(model, path):
    path = Path(path)

    with path.open("rb") as file:
        parameters = pickle.load(file)

    model_parameters = model.parameters()

    if len(parameters) != len(model_parameters):
        raise ValueError(
            "Checkpoint parameters do not match model parameters"
        )

    for parameter, saved_data in zip(
        model_parameters,
        parameters
    ):
        parameter.data = saved_data
