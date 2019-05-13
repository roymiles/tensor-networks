""" Define interfaces for a network and the weights inside the network """
from Layers.core import *
from abc import abstractmethod
from enum import Enum


class LayerTypes(Enum):
    CONV = 1
    FC = 2
    BN = 3


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

    @abstractmethod
    def get_layer_weights(self, layer_idx):
        """ Return the weights for a given layer """
        pass

    @abstractmethod
    def set_layer_weights(self, layer_idx):
        """ Set the weights for a given layer """
        pass


class INetwork:
    def __init__(self):
        raise Exception("You cannot instantiate me")

    @abstractmethod
    def set_weights(self, weights):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def num_layers(self):
        pass

    @abstractmethod
    def num_parameters(self):
        pass

    @staticmethod
    def run_layer(layer, input, **kwargs):
        """ Input is compulsory """

        # If the child classes have not overridden the behaviour, just call them with all the same arguments
        return layer(input=input, **kwargs)
