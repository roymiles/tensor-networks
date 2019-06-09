""" Use core tensors to parametrise the layers

    This is mainly experimenting with arbitrary tensor networks

"""
from Layers.layer import LayerTypes
from Layers.impl.core import *
from base import *
from Networks.network import Weights, INetwork
from Networks.graph import Graph
import Networks.Layers.core as nl
import Networks.Layers.sandbox as nls


class TuckerNet(INetwork):

    def __init__(self, architecture):
        """ Build an example tensor network
            The constructor effectively just initializes all the
            core tensors used in the tensor network

        Parameters
        ----------
        architecture : Contains the underlying architecture.
                       This is an object of type Architecture (see architectures.py)
        """

        super(TuckerNet, self).__init__()
        self.set_architecture(architecture)
        self._is_built = False
        self._switch_list = [1]

    def build(self, conv_ranks, fc_ranks, name, switch_list=[1.0]):
        """ Create/initialise the tf.Variable weights from ranks for ** ALL ** the layers
            This is not called if we are setting the weights manually

        Parameters
        ----------
        conv_ranks  : List of ranks (the same for every layer) {0:[r1, r2, r3], ... n:[...]}
        fc_ranks    : Same as above, but for fully connected layers {0:[r1, r2], ... n:[...]}
        name        : Tensorflow variable scope
        switch_list : List of all the switches with independent batch normalisation layers
                      By default we only use one switch (global batch normalisation parameters)

        The keys are the layer_idx

        Number of layers can be inferred from the network configuration
        """

        assert not self._is_built, "You cannot build the same Network more than once."

        with tf.variable_scope(name):

            # A container for all the core tensors, bn variables etc
            self._weights = Weights()

            # Need to keep track for calling the net
            self._switch_list = tf.constant(switch_list)

            for layer_idx in range(self.get_num_layers()):

                # Only need to initialize tensors for layers that have weights
                cur_layer = self.get_architecture().get_layer(layer_idx)

                # Pointwise convolution

                if isinstance(cur_layer, ConvLayer):
                    tf_weights = nls.convolution(cur_layer, layer_idx)
                    self._weights.set_conv_layer_weights(layer_idx, **tf_weights)

                elif isinstance(cur_layer, DepthwiseConvLayer):
                    # Standard depthwise layer
                    tf_weights = nl.depthwise_convolution(cur_layer, layer_idx)
                    self._weights.set_conv_layer_weights(layer_idx, **tf_weights)

                elif isinstance(cur_layer, FullyConnectedLayer):
                    tf_weights = nls.fully_connected(cur_layer, layer_idx)
                    self._weights.set_fc_layer_weights(layer_idx, **tf_weights)

                elif isinstance(cur_layer, BatchNormalisationLayer):
                    tf_weights = nl.switchable_bn(cur_layer, layer_idx, switch_list)
                    # Unpack the argument list from the tf_weights dictionary
                    self._weights.set_bn_layer_weights(layer_idx, **tf_weights)

        # All the tf Variables have been created
        self._is_built = True

    def run_layer(self, input, switch_idx, layer_idx, name):
        """ Pass input through a single layer
            Operation is dependant on the layer type

        Parameters
        ----------
        input : input is a 4 dimensional feature map [B, W, H, C]
        switch_idx: Index for switch_list
        layer_idx : Layer number
        name : For variable scoping
        """

        assert self._is_built, "The network must be built before you can run the layers."

        # Get the switch value
        switch = self._switch_list[switch_idx]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            # Merge core tensors using tensor contraction
            # Note: we take slices of the core tensors by layer idx
            # The spatial (w_h) core tensor must be merged with the core tensor

            cur_layer = self._architecture.get_layer(layer_idx)
            if isinstance(cur_layer, ConvLayer):

                # Convolutional and fully connected layers use switches
                # assert 0 < switch <= 1, "Switch must be in the range (0, 1]"

                # Combine the core tensors
                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.CONV, "The layer weights don't match up with the layer type"
                c = w["kernel"].combine(switch=switch, reshape=["W", "H", "C", "N"])

                if cur_layer.using_bias():
                    b = w["bias"].combine()
                else:
                    b = None

                # Call the function and return the result
                return cur_layer(input=input, kernel=c, bias=b)

            elif isinstance(cur_layer, FullyConnectedLayer):

                # assert 0 < switch <= 1, "Switch must be in the range (0, 1]"

                # Combine the core tensors
                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.FC, "The layer weights don't match up with the layer type"
                c = w["kernel"].combine(switch=switch, reshape=["I", "O"])

                if cur_layer.using_bias():
                    b = w["bias"].combine()
                else:
                    b = None

                return cur_layer(input=input, kernel=c, bias=b)

            elif isinstance(cur_layer, BatchNormalisationLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.BN, "The layer weights don't match up with the layer type"

                return cur_layer(input=input, mean=w["mean"][switch_idx], variance=w["variance"][switch_idx],
                                 offset=w["offset"][switch_idx], scale=w["scale"][switch_idx])
            else:
                # These layers are not overridden
                return INetwork.run_layer(cur_layer, input=input)

    def __call__(self, input, switch_idx=0):
        """ Complete forward pass for the entire network

            :param input: The input to the network e.g. a batch of images
            :param switch_idx: Index for switch_list, controls the compression of the network
                               (default, just call first switch)
        """

        # Loop through all the layers
        net = input
        for n in range(self.get_num_layers()):
            net = self.run_layer(net, switch_idx=switch_idx, layer_idx=n, name="layer_{}".format(n))

        return net
