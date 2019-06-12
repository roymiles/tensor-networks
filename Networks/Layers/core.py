"""
    These utility functions return the tf.Variables used for a given layer
    They can be used in different tensor networks.

    This is because most tensor network implementations use common structure for some layers, such as batch norm

    None of these layers should use Graph, as the tensors are not decomposed. Instead tf.get_variable is used directly.
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

    kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1], shape[2], shape[3]],
                             collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                             initializer=cur_layer.kernel_initializer,
                             regularizer=cur_layer.kernel_regularizer)

    bias = None
    if cur_layer.using_bias():
        bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[2] * shape[3]],  # W x H x C x M
                               initializer=cur_layer.bias_initializer,
                               regularizer=cur_layer.bias_regularizer,
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

    return {"kernel": kernel, "bias": bias}


def fully_connected(cur_layer, layer_idx):
    shape = cur_layer.get_shape()

    kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1]],
                             collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                             initializer=cur_layer.kernel_initializer,
                             regularizer=cur_layer.kernel_regularizer)

    bias = None
    if cur_layer.using_bias():
        bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[1]],  # I x O
                               initializer=cur_layer.bias_initializer,
                               regularizer=cur_layer.bias_regularizer,
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

    return {"kernel": kernel, "bias": bias}


def convolution(cur_layer, layer_idx):
    shape = cur_layer.get_shape()

    kernel = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1], shape[2], shape[3]],
                             collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                             initializer=tf.contrib.layers.xavier_initializer())
                             # initializer=cur_layer.kernel_initializer,
                             # regularizer=cur_layer.kernel_regularizer)

    tf.summary.histogram(f"conv_{layer_idx}", kernel)

    bias = None
    if cur_layer.using_bias():
        bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x *N*
                               # initializer=cur_layer.bias_initializer,
                               initializer=tf.initializers.zeros(),
                               regularizer=cur_layer.bias_regularizer,
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

        tf.summary.histogram(f"conv_bias_{layer_idx}", bias)

    return {"kernel": kernel, "bias": bias}
