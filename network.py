""" Abstract class for all network types """
import tensorflow as tf
from layers import *


class IWeights:
    def __init__(self):
        raise Exception("You cannot instantiate me")

    @staticmethod
    def num_parameters():
        # TODO: DO it
        return 0


class INetwork:
    def __init__(self):
        raise Exception("You cannot instantiate me")

    @staticmethod
    def run_layer(layer, input):

        # If the child classes have not overridden the behaviour, just call them
        return layer(input)

        # raise Exception("Invalid layer type: {}".format(layer))
