""" Module for the models.
"""
# standard
import os
import pickle

# external
import fbprophet

# internal
from .configation import config


def create_model(model_name, **hyperparameters):
    if model_name == "fbprophet":
        model = fbprophet.Prophet(**hyperparameters)
        model.model_name = model_name
        return model
    else:
        raise ValueError(f"Unknown model {model_name}")


def save_model(model, filename):
    filepath = os.path.join(
        config.root_path,
        "model",
        filename,
    )
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return filepath


def load_model(filename):
    filepath = os.path.join(
        config.root_path,
        "model",
        filename,
    )
    with open(filepath, "rb") as f:
        return pickle.load(f)
