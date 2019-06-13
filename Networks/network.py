""" Define interfaces for a network and the weights inside the network """
from abc import abstractmethod
from Architectures.architectures import IArchitecture
from collections import namedtuple
import tensorflow as tf


class Weights:
    """
        All networks use this weight data structure to store and query the weights values
        The weight values are either stored as tensor networks or tf.Variables
    """

    # Stored as key:value pair, where key is a layer_idx
    _weights = {}

    # All the types of weights
    Convolution = namedtuple('Convolution', ["kernel", "bias"])
    DepthwiseConvolution = namedtuple('DepthwiseConvolution', ["kernel", "bias"])
    FullyConnected = namedtuple('FullyConnected', ["kernel", "bias"])
    Mobilenetv2Bottleneck = namedtuple('Mobilenetv2Bottleneck', ["expansion_kernel", "expansion_bias",
                                                                 "depthwise_kernel", "depthwise_bias",
                                                                 "projection_kernel", "projection_bias"])

    def __init__(self):
        pass

    def set_weights(self, layer_idx, tf_weights):
        self._weights[layer_idx] = tf_weights

    def get_layer_weights(self, layer_idx):
        """
            Return the weights for a given layer
            Loops through all members in the weight namedtuple and combines them if of Graph type
        """
        w = self._weights[layer_idx]
        # for name, value in w._asdict().iteritems():
        return w

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
        NOTE: There is no setter, this value is inferred when set_architecture is called

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

    def build(self, name):
        """
            Build the tf.Variable weights used by the network

            :param name: Variable scope e.g. StandardNetwork1
        """
        with tf.variable_scope(name):

            # All the weights of the network are stored in this container
            self._weights = Weights()

            # Initialize the standard convolutional and fully connected weights
            for layer_idx in range(self._num_layers):

                # Only need to initialize tensors for layers that have weights
                cur_layer = self.get_architecture().get_layer(layer_idx)

                # If the current layer has a create_weights function, call it and add the weights
                # to the set of weights
                create_op = getattr(cur_layer, "create_weights", None)
                if callable(create_op):
                    tf_weights = create_op()(cur_layer, layer_idx)
                    self._weights.set_weights(layer_idx, tf_weights)

    def __call__(self, input, is_training=True, switch_idx=0):
        """ Complete forward pass for the entire network

            :param input: The input to the network e.g. a batch of images
            :param switch_idx: Index for switch_list, controls the compression of the network
                               (default, just call first switch)
            :param is_training: bool, is training or testing mode
        """

        tf.summary.image("Input data", input)

        # Loop through all the layers
        net = input
        for n in range(self.get_num_layers()):
            net = self.run_layer(input=net, layer_idx=n, name=f"layer_{n}",
                                 is_training=is_training, switch_idx=switch_idx)

        return net

    @staticmethod
    @abstractmethod
    def run_layer(layer, **kwargs):
        # If the child classes have not overridden the behaviour, just call them with all the same arguments
        return layer(**kwargs)