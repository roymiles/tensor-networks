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

                    tensor_network = Graph("conv_{}".format(layer_idx))

                    # A standard network just has a single high dimensional node
                    tensor_network.add_node("WHCN", shape=[shape[0], shape[1], shape[2], shape[3]],
                                            names=["W", "H", "C", "N"])

                    # Compile/generate the tf.Variables and add to the set of weights
                    tensor_network.compile()
                    kernel = tensor_network

                    if cur_layer.using_bias():
                        bias = tf.get_variable('bias_{}'.format(layer_idx),
                                               shape=shape[3],  # W x H x C x *N*
                                               initializer=tf.zeros_initializer())

                        self._weights.set_conv_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_conv_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    tensor_network = Graph("fc_{}".format(layer_idx))

                    # Create single node, compile the graph and then add to the set of weights
                    tensor_network.add_node("IO", shape=[shape[0], shape[1]], names=["I", "O"]).compile()
                    kernel = tensor_network

                    if cur_layer.using_bias():
                        bias = tf.get_variable('bias_{}'.format(layer_idx), shape=shape[1],  # I x O
                                               initializer=tf.zeros_initializer())

                        self._weights.set_fc_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_fc_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, BatchNormalisationLayer):
                    # num_features is effectively the depth of the input feature map
                    num_features = cur_layer.get_num_features()

                    # Create the mean and variance weights
                    mean = tf.get_variable('mean_{}'.format(layer_idx), shape=num_features,
                                           initializer=tf.zeros_initializer())
                    variance = tf.get_variable('variance_{}'.format(layer_idx), shape=num_features,
                                               initializer=tf.ones_initializer())

                    # When NOT affine
                    if not cur_layer.is_affine():
                        scale = None  # gamma
                        offset = None  # beta
                    else:
                        # Scale (gamma) and offset (beta) parameters
                        scale = tf.get_variable('scale_{}'.format(layer_idx), shape=num_features,
                                                initializer=tf.ones_initializer())
                        offset = tf.get_variable('offset_{}'.format(layer_idx), shape=num_features,
                                                 initializer=tf.zeros_initializer())

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
                features = cur_layer(input, kernel=w["kernel"], bias=w["bias"])
                return features
            elif isinstance(cur_layer, FullyConnectedLayer):
                w = self._weights.get_layer_weights(layer_idx)
                return cur_layer(input, kernel=w["kernel"], bias=w["bias"])
            elif isinstance(cur_layer, BatchNormalisationLayer):
                w = self._weights.get_layer_weights(layer_idx)
                return cur_layer(input, w["mean"], w["variance"], w["scale"], w["offset"])
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
