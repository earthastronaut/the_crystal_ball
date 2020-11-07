""" Module for the models.
"""
# standard
from collections import OrderedDict
import os
import pickle
import importlib

# external
import fbprophet
import pandas as pd
import pygam

# internal
from .configuration import config


class LinearGAM(pygam.LinearGAM):
    """Light wrapper around pygam.LinearGAM which is pandas aware."""

    @staticmethod
    def _feature_hyperparameters_to_arg(feature_hyperparameters: OrderedDict):
        """Convert the feature_hyperparameters to an arg for pygam.LinearGam"""
        if not isinstance(feature_hyperparameters, OrderedDict):
            raise TypeError(
                f"feature hyperparameters should be OrderedDict with keys the columns "
                f"and values the parameters. Not {type(feature_hyperparameters)}"
            )
        arg = None
        for i, feature_name in enumerate(feature_hyperparameters):
            params = feature_hyperparameters[feature_name].copy()            
            if params.get("fit_splines", True):
                tmp = pygam.s(
                    i,
                    lam=params["lam"],
                    spline_order=params["spline_order"],
                    n_splines=params["n_splines"],
                    dtype=params["dtype"],
                )
            else:
                tmp = pygam.l(
                    i, lam=params["lam"]
                )
            arg = tmp if arg is None else arg + tmp
        return arg

    def __init__(
        self,
        feature_hyperparameters: OrderedDict,
        target_column: str,
    ):
        self.target_column = target_column
        self.feature_hyperparameters = feature_hyperparameters
        arg = self._feature_hyperparameters_to_arg(feature_hyperparameters)
        super().__init__(arg)

    def get_features(self):
        """Returns a list of features for this model"""
        return list(self.feature_hyperparameters.keys())

    def transform_train(self, X_features):
        """Returns X, y for training """
        X = self.transform(X_features)
        X['_y'] = X_features.loc[X.index, self.target_column]
        X = X.dropna()
        y = X.pop('_y')
        return X, y
        
    def transform(self, X_features):
        """Returns a X matrix with only selected features"""
        features = list(self.get_features())
        try:
            X = X_features[features]
        except KeyError as error:
            errmsg = str(error) + (
                "\n\nFitting model with features.\n\n"
                f"model features (n={len(features)}): {features}\n\n"
                f"data features (n={len(X_features.columns)}): {list(X_features.columns)}\n"
            )
            raise KeyError(errmsg) from error
        return X.dropna()

    def fit(self, df_features):
        """Select from df_features the training set and return """
        X, y = self.transform_train(df_features)
        return super().fit(X, y)

    def predict(self, df_features, quantiles=(0.1, 0.5, 0.9)):
        """Select features to predict """
        if quantiles:
            return self.prediction_intervals(df_features, quantiles=quantiles)
        else:
            X = self.transform(df_features)
            return super().predict(X)

    def prediction_intervals(self, df_features, quantiles=(0.1, 0.5, 0.9)):
        """Select features to predict """
        X = self.transform(df_features)
        predict = super().prediction_intervals(X, quantiles=quantiles)

        predict = pd.DataFrame(predict, columns=quantiles)
        predict['ds'] = df_features['ds'].values
        predict['yhat'] = predict[0.5]
        predict['yhat_lower'] = predict[0.1]
        predict['yhat_upper'] = predict[0.9]
        return predict


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
