"""
    Standard weights
"""
import tensorflow as tf
from Weights.weights import Weights
from Networks.graph import Graph


def depthwise_convolution(cur_layer, layer_idx):
    """
        Implements standard depthwise convolution using a tensor network
        with a single node (aka no factorisation)
        W x H x C x M, Where M is the channel multiplier
    """

    with tf.variable_scope("DepthwiseConvolution"):
        # Very similar to standard convolution
        shape = cur_layer.get_shape()

        kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1], shape[2], shape[3]],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                 initializer=cur_layer.kernel_initializer,
                                 regularizer=cur_layer.kernel_regularizer,
                                 trainable=True)

        tf.summary.histogram(f"kernel_{layer_idx}", kernel, collections=['train'])

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[2] * shape[3]],  # W x H x C x M
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.DepthwiseConvolution(kernel, bias)


def fully_connected(cur_layer, layer_idx):
    with tf.variable_scope("FullyConnected"):
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

            tf.summary.histogram(f"bias_{layer_idx}", bias, collections=['train'])

        return Weights.FullyConnected(kernel, bias)


def convolution(cur_layer, layer_idx):
    with tf.variable_scope("Convolution"):
        shape = cur_layer.get_shape()

        kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1], shape[2], shape[3]],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                 initializer=cur_layer.kernel_initializer,
                                 regularizer=cur_layer.kernel_regularizer,
                                 trainable=True)

        k = tf.reshape(kernel, shape=(shape[0], shape[2], shape[3], shape[1]))
        tf.summary.image(f"k_{layer_idx}", k, collections=['train'])

        # tf.summary.histogram(f"conv_{layer_idx}", kernel)

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x *N*
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

            # tf.summary.histogram(f"conv_bias_{layer_idx}", bias)

        return Weights.Convolution(kernel, bias)


def mobilenetv2_bottleneck(cur_layer, layer_idx):
    """
        Standard MobileNetV2 bottleneck layer (expansion, depthwise, linear projection and residual add)
    """
    with tf.variable_scope("MobileNetV2Bottleneck"):
        weight_decay = 0.00004
        expansion = cur_layer.get_t()  # Expansion

        output_filters = cur_layer.get_c()  # Number of output channels
        input_filters = cur_layer.get_k()  # Number of input channels

        # Standard MobileNet
        expansion_kernel = tf.get_variable(f"expansion_{layer_idx}", shape=[1, 1, input_filters, input_filters*expansion],
                                           collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                           initializer=tf.keras.initializers.glorot_normal(),
                                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                           trainable=True)

        depthwise_kernel = tf.get_variable(f"depthwise_{layer_idx}", shape=[3, 3, input_filters*expansion, 1],
                                           collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                           initializer=tf.keras.initializers.glorot_normal(),
                                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                           trainable=True)

        projection_kernel = tf.get_variable(f"projection_{layer_idx}",
                                            shape=[1, 1, input_filters*expansion, output_filters],
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                            initializer=tf.keras.initializers.glorot_normal(),
                                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                            trainable=True)

        # tf.summary.histogram("expansion_kernel", expansion_kernel)
        # tf.summary.histogram("projection_kernel", projection_kernel)

        return Weights.Mobilenetv2Bottleneck(expansion_kernel, depthwise_kernel, projection_kernel)
