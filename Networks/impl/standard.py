""" Just call the layers normally, no core tensors, tensor contraction etc """
from Networks.network import INetwork, IWeights
from Layers.layer import LayerTypes
from Layers.impl.core import *
import Layers.impl.keynet as KeyNetLayers
import tensorflow as tf
from base import *


class Weights(IWeights):

    # NOTE: These are dictionaries where the index is the layer_idx
    conv = {}
    fc = {}
    bias = {}

    # Batch normalisation variables
    bn_mean = {}
    bn_variance = {}
    bn_scale = {}
    bn_offset = {}

    def __init__(self):
        pass

    def num_parameters(self):
        weight_list = []
        for dict in [self.conv, self.fc, self.bias, self.bn_mean, self.bn_variance, self.bn_scale, self.bn_offset]:
            for layer_idx, variable in dict.items():
                weight_list.append(variable)

        return IWeights.num_parameters(weight_list)

    """ The weights are inferred from their argument name """
    def set_conv_layer_weights(self, layer_idx, conv, bias):
        self.conv[layer_idx] = conv
        self.bias[layer_idx] = bias

    def set_fc_layer_weights(self, layer_idx, fc, bias):
        self.fc[layer_idx] = fc
        self.bias[layer_idx] = bias

    def set_bn_layer_weights(self, layer_idx, mean, variance, scale, offset):
        self.bn_mean[layer_idx] = mean
        self.bn_variance[layer_idx] = variance
        self.bn_scale[layer_idx] = scale
        self.bn_offset[layer_idx] = offset

    def get_layer_weights(self, layer_idx):
        """ Get the weights for this given layer.
            NOTE: This is not the cleanest way of doing this
            Currently just checks if keys exist """
        if layer_idx in self.conv:
            return {"__type__": LayerTypes.CONV, "kernel": self.conv[layer_idx], "bias": self.bias[layer_idx]}
        elif layer_idx in self.fc:
            return {"__type__": LayerTypes.FC, "kernel": self.fc[layer_idx], "bias": self.bias[layer_idx]}
        elif layer_idx in self.bn_mean:
            return {"__type__": LayerTypes.BN, "mean": self.bn_mean[layer_idx], "variance": self.bn_variance[layer_idx],
                    "scale": self.bn_scale[layer_idx], "offset": self.bn_offset[layer_idx]}
        else:
            # No weights for this layer
            return None


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

                    kernel = tf.get_variable('conv_{}'.format(layer_idx), shape=shape,
                                             initializer=tf.glorot_normal_initializer(), regularizer=l2_reg)
                    bias = tf.get_variable('bias_{}'.format(layer_idx), shape=shape[3],  # W x H x C x N
                                           initializer=tf.zeros_initializer())

                    # tf.summary.histogram("conv_kernel_{}".format(layer_idx), kernel)
                    # tf.summary.histogram("conv_bias_{}".format(layer_idx), bias)

                    self._weights.set_conv_layer_weights(layer_idx=layer_idx, conv=kernel, bias=bias)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    # Exactly the same as convolutional layer (pretty much)
                    shape = cur_layer.get_shape()
                    kernel = tf.get_variable('fc_{}'.format(layer_idx), shape=shape, initializer=initializer)
                    bias = tf.get_variable('bias_{}'.format(layer_idx), shape=shape[1],  # I x O (Except here)
                                           initializer=initializer)

                    self._weights.set_fc_layer_weights(layer_idx=layer_idx, fc=kernel, bias=bias)

                elif isinstance(cur_layer, BatchNormalisationLayer):
                    # num_features is effectively the depth of the input feature map
                    num_features = cur_layer.get_num_features()

                    # Create the mean and variance weights
                    mean = tf.get_variable('mean_{}'.format(layer_idx), shape=num_features, initializer=initializer)
                    variance = tf.get_variable('variance_{}'.format(layer_idx), shape=num_features, initializer=initializer)

                    if cur_layer.is_affine():
                        scale = None
                        offset = None
                    else:
                        # Scale (gamma) and offset (beta) parameters
                        scale = tf.get_variable('scale_{}'.format(layer_idx), shape=num_features, initializer=initializer)
                        offset = tf.get_variable('offset_{}'.format(layer_idx), shape=num_features, initializer=initializer)

                    self._weights.set_bn_layer_weights(layer_idx=layer_idx, mean=mean, variance=variance, scale=scale,
                                                       offset=offset)

    def run_layer(self, input, layer_idx, name, **kwargs):
        """ Pass input through a single layer
            Operation is dependant on the layer type

            This class is boring and doesn't add any features

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
            # NOTE: Will be None if this layer has no weights
            w = self._weights.get_layer_weights(layer_idx)
            if isinstance(cur_layer, ConvLayer):
                features = cur_layer(input, kernel=w["kernel"], bias=w["bias"])
                return features
            elif isinstance(cur_layer, FullyConnectedLayer):
                return cur_layer(input, kernel=w["kernel"], bias=w["bias"])
            elif isinstance(cur_layer, BatchNormalisationLayer):
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

        # Loop through all the layers
        net = kwargs['input']
        tf.summary.image("input", net)
        del kwargs['input']  # Don't want input in kwargs
        for n in range(self.get_num_layers()):
            net = self.run_layer(input=net, layer_idx=n, name="layer_{}".format(n), **kwargs)

        return net
