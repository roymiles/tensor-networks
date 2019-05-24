""" Define interfaces for a network and the weights inside the network """
from abc import abstractmethod
from Architectures.architectures import IArchitecture
from base import tfvar_size
from Layers.layer import LayerTypes
from Networks.graph import Graph


class Weights:
    """
        All networks use this weight data structure to store and query the weights values
        The weight values are either stored as tensor networks or tf.Variables
    """

    # Stored as key:value pair, where key is a layer_idx
    _weights = {}

    def __init__(self):
        pass

    def set_conv_layer_weights(self, layer_idx, kernel, bias):
        """
        Add a set of weights for a convolutional layer

        :param layer_idx:
        :param kernel: Is either a tf.Variable or a graph (when using tensor networks)
        :param bias: Always a tf.Variables
        :return:
        """
        self._weights[layer_idx] = {
            "__type__": LayerTypes.CONV,
            "kernel": kernel,
            "bias": bias
        }

    def set_fc_layer_weights(self, layer_idx, kernel, bias):
        """
        Add a set of weights for a fully connected layer

        :param layer_idx:
        :param kernel: Is either a tf.Variable or a graph (when using tensor networks)
        :param bias: Always a tf.Variables
        :return:
        """
        self._weights[layer_idx] = {
            "__type__": LayerTypes.FC,
            "kernel": kernel,
            "bias": bias
        }

    def set_bn_layer_weights(self, layer_idx, mean, variance, scale, offset):
        self._weights[layer_idx] = {
            "__type__": LayerTypes.BN,
            "mean": mean,
            "variance": variance,
            "scale": scale,
            "offset": offset
        }

    def num_parameters(self):
        """" Calculates the number of parameters in the weights """

        num_params = 0
        for w in self._weights.values():

            # The same approach for convolutional or fully connected weights
            if w["__type__"] == LayerTypes.CONV or w["__type__"] == LayerTypes.FC:

                if isinstance(w["kernel"], Graph):
                    # Tensor network
                    num_params += w["kernel"].num_parameters()
                else:
                    # tf.Variable
                    num_params += tfvar_size(w["kernel"])

            elif w["__type__"] == LayerTypes.BN:

                # Do not currently support tensor networks for batch norm layers
                num_params += tfvar_size(w["mean"])
                num_params += tfvar_size(w["variance"])
                num_params += tfvar_size(w["scale"])
                num_params += tfvar_size(w["offset"])

            else:
                raise Exception("Unknown weight type")

        return num_params

    def get_layer_weights(self, layer_idx):
        """ Return the weights for a given layer """
        return self._weights[layer_idx]

    def debug(self):
        for layer_idx, weight in self._weights.items():
            print("Layer {} -> {}".format(layer_idx, weight))


class INetwork:

    def __init__(self):
        # The following define the state which is common across all networks
        self._architecture = None
        self._num_layers = None
        self._weights = None

    def set_weights(self, weights):

        assert isinstance(weights, Weights), "weights must be of type IWeights"
        self._weights = weights

    def get_weights(self):
        """

        :return: IWeights: Return all the weights for the network
        """
        return self._weights

    def get_num_layers(self):
        """
        NOTE: There is not setter, this value is inferred when set_architecture is called

        :return: The number of layers in the architecture
        """
        return self._num_layers

    def set_architecture(self, architecture):
        """
        Set the architecture for the network

        :param architecture: Class of type IArchitecture
        """

        assert isinstance(architecture, IArchitecture), "architecture argument must be of type IArchitecture"

        self._architecture = architecture

        # So we don't have to recalculate it every time
        self._num_layers = architecture.num_layers()

    def get_architecture(self):
        """ Return the underlying architecture of this network, will be of type IArchitecture """
        return self._architecture

    def num_parameters(self):
        """ Get the total number of parameters in the network
            For example, in a Tucker network this will be the sum of all parameters in each core tensor """
        return self._weights.num_parameters()

    # The following methods must be overridden by every network

    @staticmethod
    @abstractmethod
    def run_layer(layer, **kwargs):
        # If the child classes have not overridden the behaviour, just call them with all the same arguments
        return layer(**kwargs)

    @abstractmethod
    def __call__(self):
        """ Complete forward pass for the entire network """
        raise Exception("All network classes must override a call function, how else do you do inference!")