"""
    These utility functions return the tf.Variables used for a given layer
    They can be used in different tensor networks.

    This is because most tensor network implementations use common structure for some layers, such as batch norm
"""

import tensorflow as tf
from Networks.graph import Graph


def switchable_batch_normalisation(cur_layer, layer_idx, switch_list=None):
    """
    Returns the weights for a (switchable?) batch normalisation layer

    :param cur_layer: Current layer, type ILayer
    :param layer_idx: Current layer id, used for assigning context/variable names
    :param switch_list: The switch list. if switches are not used, this is set to [1.] (aka only one set of BN params)
    :return: Returns the weights as a dictionary
    """
    # Implements switchable batch norm layers
    # NOTE: We don't bother using tensor networks for the batch norm layers here
    # num_features is effectively the depth of the input feature map
    num_features = cur_layer.get_num_features()

    # Independent parameters for each switch
    bn_mean = []
    bn_variance = []
    bn_scale = []
    bn_offset = []

    if not switch_list:
        switch_list = [1.]

    for switch in switch_list:
        # NOTE: Converting these to Graph is a pain because we have a stack of tf.Variables for each
        # switch and they support placeholder indexing

        # Create the mean and variance weights
        bn_mean.append(tf.get_variable(
            'mean_l{}_s{}'.format(layer_idx, switch),
            shape=num_features,
            initializer=tf.zeros_initializer())
        )

        bn_variance.append(tf.get_variable(
            'variance_l{}_s{}'.format(layer_idx, switch),
            shape=num_features,
            initializer=tf.ones_initializer())
        )

        if cur_layer.is_affine():

            # If set to None, this is equivalent to gamma=1, beta=0 (aka ignored)
            bn_scale.append(None)
            bn_offset.append(None)

        else:
            # Scale (gamma) and offset (beta) parameters
            bn_scale.append(tf.get_variable(
                'scale_l{}_s{}'.format(layer_idx, switch),
                shape=num_features,
                initializer=tf.ones_initializer())
            )

            bn_offset.append(tf.get_variable(
                'offset_l{}_s{}'.format(layer_idx, switch),
                shape=num_features,
                initializer=tf.zeros_initializer())
            )

    # Merge the array of parameters for different switches (of the same layer) into a single tensor
    # e.g. to access first switch bn_mean[layer_idx][0]
    return {
        "mean": tf.stack(bn_mean),
        "variance": tf.stack(bn_variance),
        "scale": tf.stack(bn_scale),
        "offset": tf.stack(bn_offset)
    }


def batch_normalisation(cur_layer, layer_idx):
    """
        Standard batch normalisation
    """

    # num_features is effectively the depth of the input feature map
    num_features = cur_layer.get_num_features()

    # Create the mean and variance weights
    bn_mean = tf.get_variable(
        'mean_l{}'.format(layer_idx),
        shape=num_features,
        initializer=tf.zeros_initializer())

    bn_variance = tf.get_variable(
        'variance_l{}'.format(layer_idx),
        shape=num_features,
        initializer=tf.ones_initializer())

    if cur_layer.is_affine():

        # If set to None, this is equivalent to gamma=1, beta=0 (aka ignored)
        bn_scale = None
        bn_offset = None

    else:
        # Scale (gamma) and offset (beta) parameters
        bn_scale = tf.get_variable(
            'scale_l{}'.format(layer_idx),
            shape=num_features,
            initializer=tf.ones_initializer())

        bn_offset = tf.get_variable(
            'offset_l{}'.format(layer_idx),
            shape=num_features,
            initializer=tf.zeros_initializer())

    return {
        "mean": bn_mean,
        "variance": bn_variance,
        "scale": bn_scale,
        "offset": bn_offset
    }


def depthwise_convolution(cur_layer, layer_idx):
    """
        Implements standard depthwise convolution using a tensor network
        with a single node (aka no factorisation)
    """

    # Very similar to standard convolution
    shape = cur_layer.get_shape()
    kernel = Graph("dwconv_{}".format(layer_idx))
    kernel.add_node("WHCM", shape=[shape[0], shape[1], shape[2], shape[3]],
                    names=["W", "H", "C", "M"], initializer=cur_layer.kernel_initializer,
                    regularizer=cur_layer.kernel_regularizer,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS]).compile()

    # Stay None if no bias term
    bias = None
    if cur_layer.using_bias():

        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x M
        bias.add_node("B", shape=[shape[2] * shape[3]], names=["B"],
                      initializer=cur_layer.bias_initializer, regularizer=cur_layer.bias_regularizer,
                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])
        bias.compile()

    return {"kernel": kernel, "bias": bias}


def fully_connected(cur_layer, layer_idx):
    shape = cur_layer.get_shape()

    # Create single node, compile the graph and then add to the set of weights
    kernel = Graph("fc_{}".format(layer_idx))
    kernel.add_node("IO", shape=[shape[0], shape[1]], names=["I", "O"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS]).compile()

    bias = None
    if cur_layer.using_bias():

        bias = Graph("bias_{}".format(layer_idx))  # I x O
        bias.add_node("B", shape=[shape[1]], names=["B"], initializer=tf.zeros_initializer(),
                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"]).compile()

    return {"kernel": kernel, "bias": bias}


def convolution(cur_layer, layer_idx):
    shape = cur_layer.get_shape()

    kernel = Graph("conv_{}".format(layer_idx))

    # A standard network just has a single high dimensional node
    kernel.add_node("WHCN", shape=[shape[0], shape[1], shape[2], shape[3]],
                    names=["W", "H", "C", "N"], initializer=cur_layer.kernel_initializer,
                    regularizer=cur_layer.kernel_regularizer,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

    # Compile/generate the tf.Variables and add to the set of weights
    kernel.compile()

    bias = None
    if cur_layer.using_bias():

        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x *N*
        bias.add_node("B", shape=[shape[3]], names=["B"], initializer=cur_layer.bias_initializer,
                      regularizer=cur_layer.bias_regularizer,
                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])
        bias.compile()

    return {"kernel": kernel, "bias": bias}


def mobilenetv2_bottleneck(cur_layer, layer_idx):
    """
        Standard MobileNetV2 bottleneck layer (expansion, depthwise, linear projection and residual add)
    """
    weight_decay = 0.00004
    t = cur_layer.get_t()  # Expansion

    # Yes, letter choice is contradictory with convention here, where C is commonly input channels
    c = cur_layer.get_c()  # Number of output channels
    k = cur_layer.get_k()  # Number of input channels

    # H x W x k -> H x W x (tk)    1 x 1 x k x tk
    expansion_kernel = Graph("bneck_expansion_{}".format(layer_idx))

    # H x W x (tk) -> H/s x W/s x (tk)     3 x 3 x tk x 1   (last dim is depth multiplier)
    depthwise_kernel = Graph("bneck_depthwise_{}".format(layer_idx))

    # H/s x W/s x (tk) -> H/s x W/s x n     1 x 1 x tk x n
    projection_kernel = Graph("bneck_projection_{}".format(layer_idx))

    factorise_pointwise_kernels = False

    if factorise_pointwise_kernels:
        # Factorising the pointwise kernels
        expansion_kernel.add_node("WH", shape=[1, 1],
                                  names=["W", "H"],
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                  initializer=tf.keras.initializers.he_normal(),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        expansion_kernel.add_node("C", shape=[k],
                                  names=["C"],
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                  initializer=tf.keras.initializers.he_normal(),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        expansion_kernel.add_node("N", shape=[t*k],
                                  names=["N"],
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                  initializer=tf.keras.initializers.he_normal(),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        expansion_kernel.add_edge("WH", "G", name="r1", length=1)
        expansion_kernel.add_edge("C", "G", name="r2", length=56)
        expansion_kernel.add_edge("N", "G", name="r3", length=56)
        expansion_kernel.compile()

        # ---
        projection_kernel.add_node("WH", shape=[1, 1],
                                   names=["W", "H"],
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                   initializer=tf.keras.initializers.he_normal(),
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        projection_kernel.add_node("C", shape=[t*k],
                                   names=["C"],
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                   initializer=tf.keras.initializers.he_normal(),
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        projection_kernel.add_node("N", shape=[c],
                                   names=["N"],
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                   initializer=tf.keras.initializers.he_normal(),
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

        projection_kernel.add_edge("WH", "G", name="r1", length=1)
        projection_kernel.add_edge("C", "G", name="r2", length=56)
        projection_kernel.add_edge("N", "G", name="r3", length=56)
        projection_kernel.compile()

        # Some tensorflow summaries for the core tensors g
        # First index, because spatial index is of size 1
        g1 = tf.reshape(expansion_kernel.get_node("G"), shape=(1, 56, 56, 1))
        g2 = tf.reshape(projection_kernel.get_node("G"), shape=(1, 56, 56, 1))

        tf.summary.image("Expansion g", g1)
        tf.summary.image("Projection g", g2)
        tf.summary.histogram("Expansion g", g1)
        tf.summary.histogram("Projection g", g2)

    else:
        # Standard MobileNet
        expansion_kernel.add_node("WHCN", shape=[1, 1, k, t*k],
                                  names=["W", "H", "C", "N"],
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                  initializer=tf.keras.initializers.he_normal(),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay)).compile()

        projection_kernel.add_node("WHCN", shape=[1, 1, t*k, c],
                                   names=["W", "H", "C", "N"],
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay)).compile()

    depthwise_kernel.add_node("WHCM", shape=[3, 3, t*k, 1],
                              names=["W", "H", "C", "M"],
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay)).compile()

    # Use biases for all the convolutions
    expansion_bias = Graph("expansion_bias_{}".format(layer_idx))  # W x H x C x N
    expansion_bias.add_node("B", shape=[t * k], names=["B"],
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES]).compile()

    depthwise_bias = Graph("depthwise_bias_{}".format(layer_idx))
    depthwise_bias.add_node("B", shape=[t * k], names=["B"],
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES]).compile()

    projection_bias = Graph("projection_bias_{}".format(layer_idx))
    projection_bias.add_node("B", shape=[c], names=["B"],
                             collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES]).compile()

    return {"expansion_kernel": expansion_kernel, "expansion_bias": expansion_bias,
            "depthwise_kernel": depthwise_kernel, "depthwise_bias": depthwise_bias,
            "projection_kernel": projection_kernel, "projection_bias": projection_bias}
