""" Functions for training
"""
# standard
# external
import pandas as pd
import sklearn.metrics

# internal


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
        raise ValueError("No training data")
    if test_mask.sum() == 0:
        raise ValueError("No test data")
    return (
        df_features.loc[train_mask],
        df_features.loc[test_mask],
    )


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
    windows = []
    while timestep < stop:
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
    train_df, test_df = test_train_split(df_features, *window)

    if isinstance(model, fbprophet.Prophet):
        if getattr(model, "history", "") is None:
            print(f"Fitting fbprophet with {len(train_df)} samples")
            model.fit(train_df)

    # elif isinstance(model, LinearGAM):
    #     if not model._is_fitted:
    #         print(f"Fitting LinearGAM with {len(train_df)}")
    #         model.fit(train_df)
    #     print("predictions from model")
    #     train_predict = model.prediction_intervals(train_df)
    #     test_predict = model.prediction_intervals(test_df)
    else:
        raise TypeError(f"Unknown model type {model.__class__.__name__}")


    print("predictions from model")
    train_predict = model.predict(train_df)
    test_predict = model.predict(test_df)

    required_predict_columns = {'yhat', 'yhat_lower', 'yhat_upper'}
    columns = set(train_predict.columns)
    missing_columns = required_predict_columns - columns
    if len(missing_columns) > 0:
        raise ValueError(f"Missing columns {missing_columns}")


    return {
        "model_name": model.model_name,
        "model": model,
        "df_features": df_features,
        "train_df": train_df,
        "train_predict": train_predict,
        "test_df": test_df,
        "test_predict": test_predict,
        "window": window,
        "metrics": {
            "mean_absolute_error": sklearn.metrics.mean_absolute_error(
                test_df['y'].values,
                test_predict['yhat'].values,
            ),
        },
    }


def training_results_to_evaulation_params(**training_results):
    evaluation_params = training_results.copy()
    return evaluation_params
