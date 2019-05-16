""" Define interfaces for a network and the weights inside the network """
from Layers.core import *
from abc import abstractmethod


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


class INetwork:

    def __init__(self):
        # The following define state which is common across all networks
        self._architecture = None
        self._num_layers = None
        self._weights = None

    def set_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights

    def set_num_layers(self, num_layers):
        self._num_layers = num_layers

    def get_num_layers(self):
        return self._num_layers

    def set_architecture(self, architecture):
        self._architecture = architecture

    def get_architecture(self):
        return self._architecture

    def num_parameters(self):
        """ Get the total number of parameters in the network
            For example, in a Tucker network this will be the sum of all parameters in each core tensor """
        return self._weights.num_parameters()

    # The following methods must be overridden by every network ...

    @staticmethod
    @abstractmethod
    def run_layer(layer, input, **kwargs):
        """ Input is compulsory """

        # If the child classes have not overridden the behaviour, just call them with all the same arguments
        return layer(input=input, **kwargs)

    @abstractmethod
    def __call__(self):
        """ Complete forward pass for the entire network """
        pass

    #@abstractmethod
    #def build(self, name):
    #    """ Build the weights used by the network """
    #    pass
