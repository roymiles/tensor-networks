"""
    These utility functions return the tf.Variables used for a given layer
    They can be used in different tensor networks.

    This is because most tensor network implementations use common structure for some layers, such as batch norm

    None of these layers should use Graph, as the tensors are not decomposed. Instead tf.get_variable is used directly.
"""

import tensorflow as tf
from Networks.graph import Graph
from Networks.network import Weights


class CreateWeights:
    @staticmethod
    def depthwiseConvolution(cur_layer, layer_idx):
        """
            Implements standard depthwise convolution using a tensor network
            with a single node (aka no factorisation)
        """

        # Very similar to standard convolution
        shape = cur_layer.get_shape()

        kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1], shape[2], shape[3]],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                 initializer=cur_layer.kernel_initializer,
                                 regularizer=cur_layer.kernel_regularizer,
                                 trainable=True)

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[2] * shape[3]],  # W x H x C x M
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.DepthwiseConvolution(kernel, bias)

    @staticmethod
    def fullyConnected(cur_layer, layer_idx):
        shape = cur_layer.get_shape()

        kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1]],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                 initializer=cur_layer.kernel_initializer,
                                 regularizer=cur_layer.kernel_regularizer,
                                 trainable=True)

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[1]],  # I x O
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.FullyConnected(kernel, bias)

    @staticmethod
    def convolution(cur_layer, layer_idx):
        shape = cur_layer.get_shape()

        kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1], shape[2], shape[3]],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                 initializer=cur_layer.kernel_initializer,
                                 regularizer=cur_layer.kernel_regularizer,
                                 trainable=True)

        tf.summary.histogram(f"conv_{layer_idx}", kernel)

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x *N*
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

            tf.summary.histogram(f"conv_bias_{layer_idx}", bias)

        return Weights.Convolution(kernel, bias)
