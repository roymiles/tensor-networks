""" Define interfaces for a network and the weights inside the network """
from abc import abstractmethod
from Architectures.architectures import IArchitecture
from base import tfvar_size


class IWeights:
    def __init__(self):
        raise Exception("This is an interface class with no state, you cannot call __init__")

    @staticmethod
    def num_parameters(weight_list):
        """" Calculates the number of parameters from a list of tf.Tensors
             Each elements consists of an arbitrary dimensional tensor """

        num_params = 0
        for weight in weight_list:
            if not weight:
                # e.g. offset/scale for affine BN
                continue

            num_params += tfvar_size(weight)

        return num_params

    @abstractmethod
    def get_layer_weights(self, layer_idx):
        """ Return the weights for a given layer """
        pass


class INetwork:

    def __init__(self):
        # The following define the state which is common across all networks
        self._architecture = None
        self._num_layers = None
        self._weights = None

    def set_weights(self, weights):

        assert isinstance(weights, IWeights), "weights must be of type IWeights"
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