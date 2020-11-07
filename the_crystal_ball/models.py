""" Module for the models.
"""
# standard
from collections import OrderedDict
import os
import pickle
import importlib

# external
import fbprophet

# internal
from .configuration import config


def create_model(model_name, model_class, hyperparameters=None):
    parts = model_class.split('.')
    module_path = '.'.join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    hyperparameters = hyperparameters or {}
    model = ModelClass(**hyperparameters)
    model.model_name = model_name
    return model


def save_model(model, filename):
    filepath = os.path.join(
        config.ROOT_PATH,
        "model",
        filename,
    )
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return filepath


def load_model(filename):
    filepath = os.path.join(
        config.ROOT_PATH,
        "model",
        filename,
    )
    with open(filepath, "rb") as f:
        return pickle.load(f)
