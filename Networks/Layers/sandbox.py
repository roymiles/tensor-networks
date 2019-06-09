"""
    Some experimental layer implementations
"""

import tensorflow as tf
from Networks.graph import Graph


def convolution(cur_layer, layer_idx, ranks):
    """
    WH, C, and N each individually share an auxilliary index with a core tensor g
    This core tensor, g, is a 3-way tensor of shape [r1, r2, r3] (as specified in ranks)

    :param ranks: The ranks (size of auxilliary indices)
    """
    shape = cur_layer.get_shape()

    kernel = Graph("conv_{}".format(layer_idx))

    # Add the nodes w/ exposed indices
    kernel.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
    kernel.add_node("C", shape=[shape[2]], names=["C"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
    kernel.add_node("N", shape=[shape[3]], names=["N"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])

    # Auxiliary indices
    kernel.add_edge("WH", "G", name="r0", length=ranks[0])
    kernel.add_edge("C", "G", name="r1", length=ranks[1])
    kernel.add_edge("N", "G", name="r2", length=ranks[2])

    # Compile/generate the tf.Variables and add to the set of weights
    kernel.compile()

    bias = None
    if cur_layer.using_bias():
        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x *N*
        bias.add_node("B", shape=[shape[3]], names=["B"], initializer=tf.zeros_initializer(),
                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])
        bias.compile()

    return {"kernel": kernel, "bias": bias}


def depthwise_convolution(cur_layer, layer_idx, ranks):
    """
        See Convolution
    """

    shape = cur_layer.get_shape()
    kernel = Graph("dwconv_{}".format(layer_idx))

    # Similar to standard convolution but with channel multiplier (M)
    # instead of output channels dimension
    kernel.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
    kernel.add_node("C", shape=[shape[2]], names=["C"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
    kernel.add_node("M", shape=[shape[3]], names=["M"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])

    kernel.add_edge("WH", "G", name="r0", length=ranks[0])
    kernel.add_edge("C", "G", name="r1", length=ranks[1])
    kernel.add_edge("M", "G", name="r2", length=ranks[2])
    kernel.compile()

    bias = None
    if cur_layer.using_bias():
        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x M
        bias.add_node("B", shape=[shape[2] * shape[3]], names=["B"],
                      initializer=tf.zeros_initializer(),
                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])  # Output channels is C x M
        bias.compile()

    return {"kernel": kernel, "bias": bias}


def fully_connected(cur_layer, layer_idx, ranks):
    """
        Connected I, O to a core tensor g (affectively a matrix)
    """
    shape = cur_layer.get_shape()

    kernel = Graph("fc_{}".format(layer_idx))

    # Nodes..
    kernel.add_node("I", shape=[shape[0]], names=["I"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
    kernel.add_node("O", shape=[shape[1]], names=["O"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])

    # Auxiliary indices
    kernel.add_edge("I", "G1", name="r0", length=ranks[0])
    kernel.add_edge("O", "G1", name="r1", length=ranks[1])

    # Compile the graph and add to the set of weights
    kernel.compile()

    bias = None
    if cur_layer.using_bias():
        bias = Graph("bias_{}".format(layer_idx))  # I x O
        bias.add_node("B", shape=[shape[1]], names=["B"], initializer=tf.zeros_initializer(),
                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"]).compile()

    return {"kernel": kernel, "bias": bias}








