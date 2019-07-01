"""
    Standard weights
"""
import tensorflow as tf
from Weights.weights import Weights


def convolution(cur_layer, layer_idx):
    with tf.variable_scope("Convolution"):
        shape = cur_layer.get_shape()

        kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1], shape[2], shape[3]],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                 initializer=cur_layer.kernel_initializer,
                                 regularizer=cur_layer.kernel_regularizer,
                                 trainable=True)

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x *N*
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.Convolution(kernel, bias)


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

            # tf.summary.histogram(f"bias_{layer_idx}", bias, collections=['train'])

        return Weights.FullyConnected(kernel, bias)


def mobilenetv2_bottleneck(cur_layer, layer_idx):
    """
        Standard MobileNetV2 bottleneck layer (expansion, depthwise, linear projection and residual add)
    """
    with tf.variable_scope("MobileNetV2Bottleneck"):
        weight_decay = cur_layer.get_weight_decay()
        expansion = cur_layer.get_expansion()  # Expansion
        output_filters = cur_layer.get_filters()  # Number of output channels
        input_filters = cur_layer.get_in_channels()  # Number of input channels

        # Standard MobileNet
        expansion_kernel = tf.get_variable(f"expansion_{layer_idx}",
                                           shape=[1, 1, input_filters, input_filters*expansion],
                                           collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                           initializer=tf.keras.initializers.glorot_normal(),
                                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                           trainable=True)

        depthwise_kernel = tf.get_variable(f"depthwise_{layer_idx}",
                                           shape=[3, 3, input_filters*expansion, 1],
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

        tf.summary.histogram("expansion_kernel", expansion_kernel)
        tf.summary.histogram("depthwise_kernel", depthwise_kernel)
        tf.summary.histogram("projection_kernel", projection_kernel)

        return Weights.Mobilenetv2Bottleneck(expansion_kernel, depthwise_kernel, projection_kernel)


def dense_block(cur_layer, layer_idx):
    pointwise_kernels = []
    conv_kernels = []
    with tf.variable_scope(f"DenseBlock_{layer_idx}"):
        in_channels = cur_layer.in_channels
        for i in range(cur_layer.num_layers):
            pointwise_kernel = tf.get_variable(f"pointwise_kernel_{layer_idx}_{i}",
                                               shape=[1, 1, in_channels, 4 * cur_layer.growth_rate],
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                               initializer=cur_layer.kernel_initializer,
                                               regularizer=cur_layer.kernel_regularizer,
                                               trainable=True)

            conv_kernel = tf.get_variable(f"conv_kernel_{layer_idx}_{i}",
                                          shape=[3, 3, 4 * cur_layer.growth_rate, cur_layer.growth_rate],
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                          initializer=cur_layer.kernel_initializer,
                                          regularizer=cur_layer.kernel_regularizer,
                                          trainable=True)
            # Add it to the list of kernels
            pointwise_kernels.append(pointwise_kernel)
            conv_kernels.append(conv_kernel)

            # Next layer is concatenation of input and output of previous layer
            in_channels += cur_layer.growth_rate

    # Tensorboard
    for i, (k1, k2) in enumerate(zip(pointwise_kernels, conv_kernels)):
        tf.summary.histogram(f"dense_block_pointwise_kernel_{layer_idx}_{i}", k1, collections=['train'])
        tf.summary.histogram(f"dense_block_conv_kernel_{layer_idx}_{i}", k2, collections=['train'])

    # Reuse this interface but each element is a list for each subsequent bottleneck
    return Weights.DenseNetConvBlock(pointwise_kernels=pointwise_kernels, conv_kernels=conv_kernels)
