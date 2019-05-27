""" Just call the layers normally, no core tensors, tensor contraction etc """
from Networks.network import Weights, INetwork
from Layers.layer import LayerTypes
from Layers.impl.core import *
import Layers.impl.keynet as KeyNetLayers
import tensorflow as tf
from base import *
from Networks.graph import Graph


class StandardNetwork(INetwork):
    def __init__(self, architecture):
        super(StandardNetwork, self).__init__()
        self.set_architecture(architecture)
        self._weights = None

    def build(self, name):
        """
            Build the tf.Variable weights used by the network

            :param
                name: Variable scope e.g. StandardNetwork1
        """
        with tf.variable_scope(name):

            # All the weights of the network are stored in this container
            self._weights = Weights()

            # Initialize the standard convolutional and fully connected weights
            for layer_idx in range(self._num_layers):

                # Only need to initialize tensors for layers that have weights
                cur_layer = self.get_architecture().get_layer(layer_idx)
                if isinstance(cur_layer, ConvLayer):  # or isinstance(cur_layer, Layers.axel.ConvLayer):

                    shape = cur_layer.get_shape()

                    # A standard network just has a single high dimensional node
                    kernel = Graph("conv_{}".format(layer_idx))
                    kernel.add_node("WHCN", shape=[shape[0], shape[1], shape[2], shape[3]],
                                    names=["W", "H", "C", "N"])
                    # Compile/generate the tf.Variables and add to the set of weights
                    kernel.compile()

                    if cur_layer.using_bias():

                        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x *N*
                        bias.add_node("B", shape=[shape[3]], names=["B"])
                        bias.compile(initializer=tf.zeros_initializer())

                        self._weights.set_conv_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_conv_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    shape = cur_layer.get_shape()

                    # Create single node, compile the graph and then add to the set of weights
                    kernel = Graph("fc_{}".format(layer_idx))
                    kernel.add_node("IO", shape=[shape[0], shape[1]], names=["I", "O"]).compile()

                    if cur_layer.using_bias():

                        bias = Graph("bias_{}".format(layer_idx))  # I x O
                        bias.add_node("B", shape=[shape[1]], names=["B"]).compile(initializer=tf.zeros_initializer())

                        self._weights.set_fc_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_fc_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, BatchNormalisationLayer):
                    # num_features is effectively the depth of the input feature map
                    num_features = cur_layer.get_num_features()

                    # Create the mean and variance weights
                    mean = Graph("mean_{}".format(layer_idx))
                    mean.add_node("M", shape=[num_features], names=["M"]).compile(initializer=tf.zeros_initializer())

                    variance = Graph("variance_{}".format(layer_idx))
                    variance.add_node("V", shape=[num_features], names=["V"]).compile(initializer=tf.ones_initializer())

                    # When NOT affine
                    if not cur_layer.is_affine():
                        scale = None  # gamma
                        offset = None  # beta
                    else:
                        # Scale (gamma) and offset (beta) parameters
                        scale = Graph("scale_{}".format(layer_idx))
                        scale.add_node("S", shape=[num_features], names=["S"])
                        scale.compile(initializer=tf.ones_initializer())

                        offset = Graph("offset_{}".format(layer_idx))
                        offset.add_node("O", shape=[num_features], names=["O"])
                        offset.compile(initializer=tf.zeros_initializer())

                    self._weights.set_bn_layer_weights(layer_idx=layer_idx, mean=mean, variance=variance, scale=scale,
                                                       offset=offset)

    def run_layer(self, input, layer_idx, name, **kwargs):
        """ Pass input through a single layer
            Operation is dependant on the layer type

        Parameters
        ----------
        COMPULSORY PARAMETERS
        input : input is a 4 dimensional feature map [B, W, H, C]
        layer_idx : Layer number
        name : For variable scoping

        Additional parameters are named in kwargs
        """

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            cur_layer = self.get_architecture().get_layer(layer_idx)
            if isinstance(cur_layer, ConvLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.CONV, "The layer weights don't match up with the layer type"

                c = w["kernel"].combine()

                if cur_layer.using_bias():
                    b = w["bias"].combine()
                else:
                    b = None

                return cur_layer(input, kernel=c, bias=b)

            elif isinstance(cur_layer, FullyConnectedLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.FC, "The layer weights don't match up with the layer type"

                c = w["kernel"].combine()
                b = w["bias"].combine()

                return cur_layer(input, kernel=c, bias=b)

            elif isinstance(cur_layer, BatchNormalisationLayer):

                w = self._weights.get_layer_weights(layer_idx)

                mean = w["mean"]
                if isinstance(mean, Graph):
                    mean = mean.combine()

                variance = w["variance"]
                if isinstance(variance, Graph):
                    variance = variance.combine()

                scale = w["scale"]
                if isinstance(scale, Graph):
                    scale = scale.combine()

                offset = w["offset"]
                if isinstance(offset, Graph):
                    offset = offset.combine()

                return cur_layer(input, mean, variance, scale, offset)
            elif isinstance(cur_layer, ReLU):
                act = INetwork.run_layer(layer=cur_layer, input=input, **kwargs)
                return act
            else:
                # These layers are not overridden
                return INetwork.run_layer(layer=cur_layer, input=input, **kwargs)

    def __call__(self, **kwargs):
        """ Complete forward pass for the entire network

            :return net: Result from forward pass"""

        self._weights.debug()

        # Loop through all the layers
        net = kwargs['input']
        tf.summary.image("input", net)
        del kwargs['input']  # Don't want input in kwargs
        for n in range(self.get_num_layers()):
            net = self.run_layer(input=net, layer_idx=n, name="layer_{}".format(n), **kwargs)

        return net
