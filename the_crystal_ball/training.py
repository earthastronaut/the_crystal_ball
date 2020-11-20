""" Functions for training
"""
# standard
import copy
import itertools
import random

# external
import fbprophet
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
from pygam.pygam import LinearGAM


def test_train_split(df_features, train_range, test_range):
    """Split the data into training and test data"""
    train_start, train_stop = train_range
    test_start, test_stop = test_range

    test_start = test_start or train_stop

    # Covert to timestamps which allows us to pass in strings
    # if we want.
    train_start = pd.Timestamp(str(train_start))
    train_stop = pd.Timestamp(str(train_stop))
    test_start = pd.Timestamp(str(test_start))
    test_stop = pd.Timestamp(str(test_stop))

    timestamps = df_features["ds"]

    train_mask = (timestamps >= train_start) & (timestamps < train_stop)
    test_mask = (timestamps >= test_start) & (timestamps < test_stop)
    if train_mask.sum() == 0:
        xmin, xmax = timestamps.iloc[[0, -1]]
        raise ValueError(
            f"No training data: ({xmin}, {xmax}) "
            f"doesnt match ({train_start}, {train_stop})"
        )
    if test_mask.sum() == 0:
        xmin, xmax = timestamps.iloc[[0, -1]]
        raise ValueError(
            f"No test data: ({xmin}, {xmax}) "
            f"doesnt match ({test_start}, {test_stop})"
        )

    (train_index,) = np.where(train_mask)
    (test_index,) = np.where(test_mask)
    return train_index, test_index


class TimeSeriesSplit:
    """ Takes the windows and returns sklearn capatible TimeSeriesSplit object """

    def __init__(self, windows):
        self.windows = windows

    def split(self, df_features):
        """ Split the features into train_index, test_index """
        for window in self.windows:
            yield test_train_split(df_features, *window)


class Scorer:
    def __init__(
        self,
        metric=sklearn.metrics.median_absolute_error,
        bigger_is_better=False,
    ):
        self.metric = metric
        self.sign = 1 if bigger_is_better else -1

    def __call__(self, estimator, test_df, sample_weight=None):
        y_pred = estimator.predict(test_df)
        return self.sign * self.metric(test_df["y"], y_pred["yhat"])


def compute_sliding_windows(
    start,
    stop,
    step_delta,
    interval_delta,
    training_size=25,
    test_size=5,
):
    """Compute the sliding windows"""
    start = pd.Timestamp(str(start))
    stop = pd.Timestamp(str(stop))
    step_delta = pd.Timedelta(step_delta)
    interval_delta = pd.Timedelta(interval_delta)

    timestep = start
    test_stop = start
    windows = []
    while test_stop < stop:
        train_start = timestep
        train_stop = train_start + interval_delta * training_size
        test_start = train_stop
        test_stop = test_start + interval_delta * test_size

        windows.append(
            (
                (train_start, train_stop),
                (test_start, test_stop),
            )
        )
        timestep += step_delta
    return windows


def compute_sliding_windows_for_data(df_features, **kws):
    """Compute sliding windows based on start/stop of features """
    timestamps = df_features["ds"]
    interval_delta = timestamps[1] - timestamps[0]
    kws.setdefault("interval_delta", interval_delta)
    kws.setdefault("training_size", 100)
    kws.setdefault("test_size", 30)

    interval_delta = kws["interval_delta"]
    test_size = kws["test_size"]

    start = kws.pop("start", timestamps.min())
    stop = kws.pop("stop", timestamps.max())
    step_delta = kws.pop("step_delta", interval_delta * int(test_size * 0.8))

    return compute_sliding_windows(start, stop, step_delta, **kws)


def train_model(model, df_features, window):
    train_index, test_index = test_train_split(df_features, *window)
    train_df = df_features.iloc[train_index]
    test_df = df_features.iloc[test_index]

    if isinstance(model, fbprophet.Prophet):
        if getattr(model, "history", "") is None:
            print(f"Fitting fbprophet with {len(train_df)} samples")
            model.fit(train_df)
    elif isinstance(model, LinearGAM):
        if not model._is_fitted:
            print(f"Fitting LinearGAM with {len(train_df)}")
            model.fit(train_df)
    else:
        raise TypeError(f"Unknown model type {model.__class__.__name__}")

    print("predictions from model")
    train_predict = model.predict(train_df)
    test_predict = model.predict(test_df)

    required_predict_columns = {"yhat", "yhat_lower", "yhat_upper"}
    columns = set(train_predict.columns)
    missing_columns = required_predict_columns - columns
    if len(missing_columns) > 0:
        raise ValueError(f"Missing columns {missing_columns}")

    return {
        "model": model,
        "df_features": df_features,
        "train_df": train_df,
        "train_predict": train_predict,
        "test_df": test_df,
        "test_predict": test_predict,
        "window": window,
        "metrics": {
            "mean_absolute_error": sklearn.metrics.mean_absolute_error(
                test_df["y"].values,
                test_predict["yhat"].values,
            ),
        },
    }


def training_results_to_evaulation_params(**training_results):
    evaluation_params = training_results.copy()
    return evaluation_params


def generate_hyperparameter_grid_linear_gam(
    base_hyperparameters, number_feature_parameters=30, feature_parameters=None
):
    """
    options = {
        'n_splines': np.linspace(1, 20, 10).astype(int),
        'spline_order': np.arange(3, 10, 5).astype(int),
        'lam': np.logspace(-3, 3, 11),
    }
    """
    grid = []
    feature_hyperparameters = base_hyperparameters["feature_hyperparameters"]
    for _ in range(number_feature_parameters):
        feature_hyperparameters_new = copy.deepcopy(feature_hyperparameters)
        for feature_name, params in feature_hyperparameters_new.items():
            while True:
                for key, values in feature_parameters.items():
                    params[key]
                    params[key] = random.choice(values)
                if params["n_splines"] <= params["spline_order"]:
                    # n_splines must be > spline_order
                    continue
                break
            feature_hyperparameters_new[feature_name] = params
        grid.append(feature_hyperparameters_new)

    param_grid = {k: [v] for k, v in base_hyperparameters.items()}
    param_grid["feature_hyperparameters"] = grid
    return param_grid


def hyperparamter_search(model, df_features, windows, **kws):
    kws_search = {
        "estimator": model,
        "n_jobs": 10,
        "cv": TimeSeriesSplit(windows).split(df_features),
        "scoring": Scorer(),
    }
    if isinstance(model, LinearGAM):
        param_grid = kws["param_grid"]
        if "base_hyperparameters" in param_grid:
            param_grid = generate_hyperparameter_grid_linear_gam(**param_grid)
            kws["param_grid"] = param_grid
    kws_search.update(kws)
    model_search = sklearn.model_selection.GridSearchCV(**kws_search)
    return model_search.fit(df_features)
