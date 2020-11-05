""" Manage configuration for the model
"""
import os


class Configuration:
    """ Class object for handling configuration """

    ROOT_PATH = os.path.abspath(os.path.join(__file__, "../.."))


config = Configuration()
