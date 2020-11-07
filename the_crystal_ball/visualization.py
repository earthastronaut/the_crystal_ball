""" Visualization functions for the model
"""
# external
import matplotlib.pylab as plt
import seaborn as sns
import fbprophet
import pandas as pd

# internal
from .training import test_train_split

# Default parameters for matplotlib
plt.rcParams["figure.figsize"] = (11, 11)

COLOR_TRAIN = sns.crayons["Denim"]
COLOR_TEST = sns.crayons["Orange"]
COLOR_PREDICT = sns.crayons["Royal Purple"]


def plot_articles(df_input, ax=None, colors=None, add_legend=True):
    ax = ax or plt.gca()

    articles = df_input["article"].unique()
    if colors is None:
        colors = sns.color_palette("rainbow", len(articles))

    colors = iter(colors)
    for article in articles:
        views = df_input.loc[df_input["article"] == article]["views"]

        # views = views / views.sum()
        views = views.rolling(10).mean()
        # views = views.cumsum()
        color = next(colors)
        plt.plot(
            views.index,
            views.values,
            label=article,
            color=color,
            lw=2,
        )

        ax.axhline(views.mean(), label=article, color=color)

    if add_legend:
        plt.legend()
    ax.set_ylabel("Daily Page Views")
    ax.set_xlabel("")


def plot_features(df_features, ax=None, **kws):
    kws.setdefault("alpha", 0.5)
    ax = ax or plt.gca()
    if isinstance(df_features, pd.Series):
        ax.plot(df_features.index, df_features.values, **kws)
    else:
        ax.plot(df_features['ds'], df_features['y'], **kws)


def plot_features_for_window(df_features, window, ax=None, **kws):
    ax = ax or plt.gca()

    train_range, test_range = window
    train_df, test_df = test_train_split(df_features, train_range, test_range)

    ax.set_title(train_range[0])
    plot_features(
        train_df,
        color=COLOR_TRAIN,
        label="Train",
        ax=ax,
        **kws,
    )
    plot_features(
        test_df,
        color=COLOR_TEST,
        label="Test",
        ax=ax,
        **kws,
    )


def figure_test_train_splits(df_features, windows, max_windows=5):
    max_windows = min(len(windows), max_windows)

    fig, axes = plt.subplots(
        nrows=max_windows,
        ncols=1,
        sharex=True,
        figsize=(20, 10),
    )
    axes = axes.ravel()
    for i in range(max_windows):
        plot_features_for_window(df_features, window=windows[i], ax=axes[i])
    plt.legend()
    fig.tight_layout()
    return fig


def plot_evaluation_model_prediction(predict_df, ax=None, **kws):
    ax = ax or plt.gca()
    kws_plot = {
        "color": COLOR_PREDICT,
        "alpha": 0.5,
    }
    kws_plot.update(kws)
    plt.plot(predict_df["ds"], predict_df["yhat"], **kws_plot)
    kws_plot.pop("label", None)
    plt.fill_between(
        predict_df["ds"],
        y1=predict_df["yhat_lower"],
        y2=predict_df["yhat_upper"],
        **kws_plot,
    )


def plot_evaluation_prediction(
    train_df,
    train_predict,
    test_df,
    test_predict,
    model,
    ax=None,
    plot_kws=None,
    **other
):
    ax = ax or plt.gca()
    plot_kws = plot_kws or {}

    plot_evaluation_model_prediction(train_predict, label="Predict", ax=ax, **plot_kws)  # noqa
    plot_evaluation_model_prediction(test_predict, ax=ax, **plot_kws)

    plot_features(
        train_df,
        color=COLOR_TRAIN,
        label="Train",
        ax=ax,
        **plot_kws,
    )
    plot_features(
        test_df,
        color=COLOR_TEST,
        label="Test",
        ax=ax,
        **plot_kws,
    )


def plot_evaluation_fbprophet_forecast_residual(
    input_df, predict_df, ax=None, **kws
):  # noqa
    ax = ax or plt.gca()

    input_y = input_df["y"].values

    kws_plot = {
        "color": COLOR_PREDICT,
        "alpha": 0.5,
    }
    kws_plot.update(kws)
    plt.plot(predict_df["ds"], predict_df["yhat"].values - input_y, **kws_plot)
    kws_plot.pop("label", None)
    plt.fill_between(
        predict_df["ds"],
        y1=predict_df["yhat_lower"].values - input_y,
        y2=predict_df["yhat_upper"].values - input_y,
        **kws_plot,
    )


def plot_evaluation_prediction_residual(
    train_df,
    train_predict,
    test_df,
    test_predict,
    model,
    ax=None,
    plot_kws=None,
    add_legend=True,
    **other
):
    ax = ax or plt.gca()
    plot_kws = plot_kws or {}

    plot_evaluation_fbprophet_forecast_residual(
        train_df,
        train_predict,
        label="Predict Train",
        color=COLOR_TRAIN,
        ax=ax,
        **plot_kws,
    )
    plot_evaluation_fbprophet_forecast_residual(
        test_df,
        test_predict,
        label="Predict Test",
        color=COLOR_TEST,
        ax=ax,
        **plot_kws,  # noqa
    )
    if add_legend:
        plt.legend()
    ax.axhline(0, color="k")
    plt.ylabel("Residuals")


def figure_evaluation_components(**training_results):
    model = training_results['model']
    model_name = training_results['model_name']
    figure = model.plot_components(
        training_results['test_predict'],
        uncertainty=True,
    )
    ax = figure.axes[0]
    ax.set_title(model_name)
    # train_df = training_results['train_df']
    test_df = training_results['test_df']
    ax.plot(
        test_df['ds'].values,
        test_df['y'].values,
    )

    return figure

 