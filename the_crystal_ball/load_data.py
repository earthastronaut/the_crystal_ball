""" Module for loading our data
"""
# standard library
import os
import json

# external
import pandas as pd

# local
import requests


def get_wikipedia_daily_page_views(article, start=None, stop=None):
    """Reads daily page_views from wikipedia for an article.

    Args:
        articles (List[str]): Names of wikipedia articles
        start (Timestamp): Timestamp for start, otherwise is not included
        stop (Timestamp): Timestamp to stop, otherwise not included. Must
            have start time.

    Returns:
        pd.DataFrame: Data frame with daily page views. Columns:
            * project
            * article
            * granularity
            * timestamp
            * access
            * agent
            * views
    """
    url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}"  # noqa
    params = {
        "project": "en.wikipedia",
        "access": "all-access",
        "agent": "all-agents",
        "article": article,
        "granularity": "daily",
    }
    if start:
        url = os.path.join(url, "{start}")
        params["start"] = pd.Timestamp(start).strftime("%Y%m%d")
        if stop:
            url = os.path.join(url, "{stop}")
            params["stop"] = pd.Timestamp(stop).strftime("%Y%m%d")

    url_with_params = url.format(**params)
    print(f"Downloading: {url_with_params}")
    response = requests.get(url_with_params)
    data = response.json()
    try:
        df = pd.DataFrame(data["items"])
        df["timestamp"] = df["timestamp"].map(lambda x: pd.Timestamp(x[:8]))
    except Exception as error:
        raise error.__class__(str(error) + "   " + json.dumps(data)) from error
    return df


def get_wikipedia_articles(articles, start=None, stop=None):
    """Download multiple articles

    Args:
        articles (List[str]): Names of wikipedia articles
        start (Timestamp): Timestamp for start, otherwise is not included
        stop (Timestamp): Timestamp to stop, otherwise not included. Must
            have start time.

    Returns:
        pd.DataFrame: Data frame with daily page views. Columns:
            * project
            * article
            * granularity
            * timestamp
            * access
            * agent
            * views

    """
    dataframes = []
    for article in articles:
        print(f"Download article: {article}")
        dataframes.append(get_wikipedia_daily_page_views(article, start, stop))

    return pd.concat(dataframes)
