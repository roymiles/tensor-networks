""" Define interfaces for a network and the weights inside the network """
import tensorflow as tf
from layers import *


class IWeights:
    def __init__(self):
        raise Exception("You cannot instantiate me")

    @staticmethod
    def num_parameters(weight_list):
        """" Calculates the number of parameters from a list of weights
             Each elements consists of an arbitrary dimensional tensor """

        num_params_op = 0
        for weight in weight_list:
            num_params_op += tf.size(weight)

        sess = tf.Session()
        num_params = sess.run(num_params_op)
        return num_params


class INetwork:
    def __init__(self):
        raise Exception("You cannot instantiate me")

    @staticmethod
    def run_layer(layer, input):

        # If the child classes have not overridden the behaviour, just call them
        return layer(input)

        # raise Exception("Invalid layer type: {}".format(layer))
