import tensorflow as tf
from Weights.weights import Weights
from Networks.graph import Graph


def convolution(cur_layer, layer_idx):
    """
        WH, C, and N each individually share an auxilliary index with a core tensor g
        This core tensor, g, is a 3-way tensor of shape [r1, r2, r3] (as specified in ranks)
    """
    shape = cur_layer.get_shape()

    # The size of auxilliary indices
    ranks = cur_layer.ranks
    assert len(ranks) == 3, "Must specified r0, r1, r2"

    kernel = Graph("conv_{}".format(layer_idx))

    # Add the nodes w/ exposed indices
    kernel.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
    kernel.add_node("C", shape=[shape[2]], names=["C"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
    kernel.add_node("N", shape=[shape[3]], names=["N"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

    # Auxiliary indices
    kernel.add_edge("WH", "G", name="r0", length=ranks[0])
    kernel.add_edge("C", "G", name="r1", length=ranks[1])
    kernel.add_edge("N", "G", name="r2", length=ranks[2])

    # Compile/generate the tf.Variables and add to the set of weights
    kernel.compile()
    kernel.set_output_shape(["W", "H", "C", "N"])

    bias = None
    if cur_layer.using_bias():
        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x *N*
        bias.add_node("B", shape=[shape[3]], names=["B"], initializer=tf.zeros_initializer(),
                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])
        bias.compile()

    return Weights.Convolution(kernel, bias)


def depthwise_convolution(cur_layer, layer_idx, ranks):
    """
        See Convolution
    """

    shape = cur_layer.get_shape()
    kernel = Graph("dwconv_{}".format(layer_idx))

    # Similar to standard convolution but with channel multiplier (M)
    # instead of output channels dimension
    kernel.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
    kernel.add_node("C", shape=[shape[2]], names=["C"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
    kernel.add_node("M", shape=[shape[3]], names=["M"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

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

    return Weights.DepthwiseConvolution(kernel, bias)


def fully_connected(cur_layer, layer_idx, ranks):
    """
        Connected I, O to a core tensor g (affectively a matrix)
    """
    shape = cur_layer.get_shape()

    kernel = Graph("fc_{}".format(layer_idx))

    # Nodes..
    kernel.add_node("I", shape=[shape[0]], names=["I"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
    kernel.add_node("O", shape=[shape[1]], names=["O"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

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

    return Weights.FullyConnected(kernel, bias)


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

    # Factorising the pointwise kernels
    expansion_kernel.add_node("WH", shape=[1, 1],
                              names=["W", "H"],
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                              initializer=tf.keras.initializers.glorot_normal(),
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    expansion_kernel.add_node("C", shape=[k],
                              names=["C"],
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                              initializer=tf.keras.initializers.glorot_normal(),
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    expansion_kernel.add_node("N", shape=[t * k],
                              names=["N"],
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                              initializer=tf.keras.initializers.glorot_normal(),
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    expansion_kernel.add_edge("WH", "G", name="r1", length=1)
    expansion_kernel.add_edge("C", "G", name="r2", length=56)
    expansion_kernel.add_edge("N", "G", name="r3", length=56)
    expansion_kernel.compile()

    # ---
    projection_kernel.add_node("WH", shape=[1, 1],
                               names=["W", "H"],
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                               initializer=tf.keras.initializers.glorot_normal(),
                               regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    projection_kernel.add_node("C", shape=[t * k],
                               names=["C"],
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                               initializer=tf.keras.initializers.glorot_normal(),
                               regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    projection_kernel.add_node("N", shape=[c],
                               names=["N"],
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                               initializer=tf.keras.initializers.glorot_normal(),
                               regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    projection_kernel.add_edge("WH", "G", name="r1", length=1)
    projection_kernel.add_edge("C", "G", name="r2", length=56)
    projection_kernel.add_edge("N", "G", name="r3", length=56)
    projection_kernel.compile()

    depthwise_kernel.add_node("WHCM", shape=[3, 3, t * k, 1],
                              names=["W", "H", "C", "M"],
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                              initializer=tf.keras.initializers.glorot_normal(),
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay)).compile()

    # Some tensorflow summaries for the core tensors g
    # First index, because spatial index is of size 1
    g1 = tf.reshape(expansion_kernel.get_node("G"), shape=(1, 56, 56, 1))
    g2 = tf.reshape(projection_kernel.get_node("G"), shape=(1, 56, 56, 1))

    tf.summary.image("Expansion g", g1)
    tf.summary.image("Projection g", g2)
    tf.summary.histogram("Expansion g", g1)
    tf.summary.histogram("Projection g", g2)

    # Use biases for all the convolutions
    expansion_bias = tf.get_variable(f"expansion_bias_{layer_idx}", shape=[t * k],
                                     initializer=tf.keras.initializers.constant(0),
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

    depthwise_bias = tf.get_variable(f"depthwise_bias_{layer_idx}", shape=[t * k],
                                     initializer=tf.keras.initializers.constant(0),
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

    projection_bias = tf.get_variable(f"projection_bias_{layer_idx}", shape=[c],
                                      initializer=tf.keras.initializers.constant(0),
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

    tf.summary.histogram("expansion_bias", expansion_bias)
    tf.summary.histogram("depthwise_bias", depthwise_bias)
    tf.summary.histogram("projection_bias", projection_bias)

    return Weights.Mobilenetv2Bottleneck(expansion_kernel, expansion_bias,
                                         depthwise_kernel, depthwise_bias,
                                         projection_kernel, projection_bias)