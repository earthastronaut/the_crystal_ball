""" Module for the models.
"""
# standard
import functools
import importlib
import os
import pickle
from collections import OrderedDict

# external
import fbprophet
import numpy as np
import pandas as pd
import pygam

# internal
from .configuration import config


def resolve_term_class(TermClass: str = "pygam.terms.SplineTerm"):
    """Resolve the term class string into a class definition. """
    module_name, _, class_name = TermClass.rpartition(".")
    if module_name == "":
        return getattr(pygam.terms, class_name)
    else:
        return getattr(importlib.import_module(module_name), class_name)


class LinearGAM(pygam.LinearGAM):
    """Light wrapper around pygam.LinearGAM which is pandas aware."""

    @staticmethod
    def _feature_hyperparameters_to_terms(feature_hyperparameters: OrderedDict):
        """Convert the feature_hyperparameters to an arg for pygam.LinearGam"""
        if not isinstance(feature_hyperparameters, OrderedDict):
            raise TypeError(
                f"feature hyperparameters should be OrderedDict with keys the columns "
                f"and values the parameters. Not {type(feature_hyperparameters)}"
            )
        terms = pygam.terms.TermList()
        for i, feature_name in enumerate(feature_hyperparameters):
            params = feature_hyperparameters[feature_name].copy()
            TermClass = resolve_term_class(params.pop("TermClass", "SplineTerm"))
            try:
                term = TermClass(
                    feature=i,
                    **params,
                )
            except Exception as error:
                errmsg = f"Error when creating {TermClass.__name__} with {params}\n"
                errmsg += str(error)
                raise error.__class__(errmsg) from error

            terms += term
        return terms

    def __init__(
        self,
        feature_hyperparameters: OrderedDict,
        target_column: str,
        model_name: str = None,
        callbacks=["deviance", "diffs"],
        **kws,
    ):
        super().__init__(**kws)

        # TODO: fixes a bug in LinearGAM where the callbacks are not passed through
        self.callbacks = callbacks

        self.model_name = model_name
        self.target_column = target_column
        self.feature_hyperparameters = feature_hyperparameters
        self.terms = self._feature_hyperparameters_to_terms(
            self.feature_hyperparameters
        )

        # terms are created from the feature_hypterparameters and X
        self._include += ["feature_hyperparameters", "target_column", "model_name"]
        self._exclude += ["terms"]

    def set_params(self, **params):
        super().set_params(**params)
        self.terms = self._feature_hyperparameters_to_terms(
            self.feature_hyperparameters
        )
        return self

    def get_features(self):
        """Returns a list of features for this model"""
        return list(self.feature_hyperparameters.keys())

    def transform_train(self, X_features):
        """Returns X, y for training """
        X = self.transform(X_features)
        X["_y"] = X_features.loc[X.index, self.target_column]
        X = X.dropna()
        y = X.pop("_y")
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
        predict["ds"] = df_features["ds"].values
        predict["yhat"] = predict[0.5]
        predict["yhat_lower"] = predict[0.1]
        predict["yhat_upper"] = predict[0.9]
        return predict

    def predict_terms(self, df_features):
        """Return prediction per term """
        X = self.transform(df_features)
        modelmat = self._modelmat(X)
        predictions = []
        i = 0
        for term in self.terms:
            columns = slice(i, i + term.n_coefs)
            y = modelmat[:, columns].dot(self.coef_[columns])
            predictions.append(y)
            i = columns.stop

        return np.row_stack(predictions)


class Prophet(fbprophet.Prophet):
    def set_params(self, **params):
        return self.__class__(**params)

    def get_params(self, deep=False):
        include = [
            "growth",
            # "changepoints",
            "n_changepoints",
            "changepoint_range",
            "yearly_seasonality",
            "weekly_seasonality",
            "daily_seasonality",
            "holidays",
            "seasonality_mode",
            "seasonality_prior_scale",
            "holidays_prior_scale",
            "changepoint_prior_scale",
            "mcmc_samples",
            "interval_width",
            "uncertainty_samples",
            # "stan_backend",
        ]
        return {k: getattr(self, k) for k in include}


def create_model(model_name, model_class, hyperparameters=None):
    module_path, _, class_name = model_class.rpartition(".")
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
