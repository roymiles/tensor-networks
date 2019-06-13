"""
    These utility functions return the tf.Variables used for a given layer
    They can be used in different tensor networks.

    This is because most tensor network implementations use common structure for some layers, such as batch norm

    None of these layers should use Graph, as the tensors are not decomposed. Instead tf.get_variable is used directly.
"""

import tensorflow as tf
from Networks.network import Weights


class CreateWeights:
    @staticmethod
    def depthwise_convolution(cur_layer, layer_idx):
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
    def fully_connected(cur_layer, layer_idx):
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

    @staticmethod
    def mobilenetv2_bottleneck(cur_layer, layer_idx):
        """
            Standard MobileNetV2 bottleneck layer (expansion, depthwise, linear projection and residual add)
        """
        weight_decay = 0.00004
        t = cur_layer.get_t()  # Expansion

        # Yes, letter choice is contradictory with convention here, where C is commonly input channels
        c = cur_layer.get_c()  # Number of output channels
        k = cur_layer.get_k()  # Number of input channels

        # Standard MobileNet
        expansion_kernel = tf.get_variable(f"expansion_{layer_idx}", shape=[1, 1, k, t*k],
                                           collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                           initializer=tf.keras.initializers.glorot_normal(),
                                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                           trainable=True)

        projection_kernel = tf.get_variable(f"projection_{layer_idx}", shape=[1, 1, t*k, c],
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                            initializer=tf.keras.initializers.glorot_normal(),
                                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                            trainable=True)

        depthwise_kernel = tf.get_variable(f"depthwise_{layer_idx}", shape=[3, 3, t*k, 1],
                                           collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                           initializer=tf.keras.initializers.glorot_normal(),
                                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                           trainable=True)

        tf.summary.histogram("expansion_kernel", expansion_kernel)
        tf.summary.histogram("projection_kernel", projection_kernel)

        # Use biases for all the convolutions
        expansion_bias = tf.get_variable(f"expansion_bias_{layer_idx}", shape=[t * k],
                                         initializer=tf.keras.initializers.constant(0),
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                         trainable=True)

        depthwise_bias = tf.get_variable(f"depthwise_bias_{layer_idx}", shape=[t * k],
                                         initializer=tf.keras.initializers.constant(0),
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                         trainable=True)

        projection_bias = tf.get_variable(f"projection_bias_{layer_idx}", shape=[c],
                                          initializer=tf.keras.initializers.constant(0),
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                          trainable=True)

        tf.summary.histogram("expansion_bias", expansion_bias)
        tf.summary.histogram("depthwise_bias", depthwise_bias)
        tf.summary.histogram("projection_bias", projection_bias)

        return Weights.Mobilenetv2Bottleneck(expansion_kernel, expansion_bias,
                                             depthwise_kernel, depthwise_bias,
                                             projection_kernel, projection_bias)
