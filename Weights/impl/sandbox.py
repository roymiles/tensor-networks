import tensorflow as tf
from Weights.weights import Weights
from Networks.graph import Graph


def convolution(cur_layer, layer_idx):
    """
        WH, C, and N each individually share an auxilliary index with a core tensor g
        This core tensor, g, is a 3-way tensor of shape [r1, r2, r3] (as specified in ranks)
    """
    with tf.variable_scope("Convolution"):
        shape = cur_layer.get_shape()

        # The size of auxilliary indices
        ranks = cur_layer.ranks
        assert len(ranks) == 3, "Must specified r0, r1, r2"

        k1 = Graph("conv_{}".format(layer_idx))

        # Ratio R that is compressed
        R = 0.8
        s1 = int(shape[3] * R)
        s2 = shape[3] - s1

        # Add the nodes w/ exposed indices
        k1.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
        k1.add_node("C", shape=[shape[2]], names=["C"],
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
        k1.add_node("N", shape=[s1], names=["N"],
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

        # Auxiliary indices
        # NOTE: Must specify shared at start
        k1.add_edge("WH", "G", name="r0", length=ranks[0], shared=True)
        k1.add_edge("C", "G", name="r1", length=ranks[1])
        k1.add_edge("N", "G", name="r2", length=ranks[2])

        # Compile/generate the tf.Variables and add to the set of weights
        k1.compile()
        k1.set_output_shape(["W", "H", "C", "N"])

        # Extra bit
        k2 = tf.get_variable(f"k2_{layer_idx}", shape=[shape[0], shape[1], shape[2], s2],
                             collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                             initializer=cur_layer.kernel_initializer,
                             regularizer=cur_layer.kernel_regularizer,
                             trainable=True)

        # Some plots for Tensorboard
        c = k1.get_node("C")
        c_shape = c.get_shape().as_list()
        c = tf.reshape(c, shape=(1, c_shape[0], c_shape[1], 1))
        tf.summary.image(f"C_{layer_idx}", c, collections=['train'])

        n = k1.get_node("N")
        n_shape = n.get_shape().as_list()
        n = tf.reshape(n, shape=(1, n_shape[0], n_shape[1], 1))
        tf.summary.image(f"N_{layer_idx}", n, collections=['train'])

        g = k1.get_node("G")
        g_shape = g.get_shape().as_list()
        g = tf.reshape(g, shape=(g_shape[0], g_shape[1], g_shape[2], 1))
        tf.summary.image(f"G_{layer_idx}", g, collections=['train'])

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x *N*
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.ConvolutionTest(k1, k2, bias)
        #return Weights.Convolution(k1, bias)


def depthwise_convolution(cur_layer, layer_idx, ranks):
    """
        See Convolution
    """
    with tf.variable_scope("DepthwiseConvolution"):
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
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[2] * shape[3]],  # W x H x C x *M*
                                   initializer=tf.zeros_initializer(),
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)  # Output channels is C x M

        return Weights.DepthwiseConvolution(kernel, bias)


def fully_connected(cur_layer, layer_idx, ranks):
    """
        Connected I, O to a core tensor g (affectively a matrix)
    """
    with tf.variable_scope("FullyConnected"):
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
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[1]],  # I x O
                                   initializer=tf.zeros_initializer(),
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.FullyConnected(kernel, bias)


def mobilenetv2_bottleneck(cur_layer, layer_idx):
    """
        Standard MobileNetV2 bottleneck layer (expansion, depthwise, linear projection and residual add)
    """

    with tf.variable_scope("MobileNetV2Bottleneck"):
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


def pointwise_dot(cur_layer, layer_idx):
    shape = cur_layer.get_shape()

    with tf.variable_scope("PointwiseDot", reuse=tf.AUTO_REUSE):
        c = tf.get_variable(f"c_{layer_idx}", shape=[shape[0], shape[1]],
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                            initializer=cur_layer.kernel_initializer,
                            regularizer=cur_layer.kernel_regularizer,
                            trainable=True)

        tf.summary.histogram(f"c_{layer_idx}", c, collections=['train'])
        _c = tf.reshape(c, shape=(1, shape[0], shape[1], 1))
        tf.summary.image("c", _c, collections=['train'])
        tf.summary.histogram(f"c", _c, collections=['train'])

        # REMEMBER: f"g_{layer_idx}" if not reusing
        g = tf.get_variable(f"g", shape=[shape[1], shape[2]],
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                            initializer=cur_layer.kernel_initializer,
                            regularizer=cur_layer.kernel_regularizer,
                            trainable=True)

        # Create a constant sine ting
        """import numpy as np
        c = np.zeros((1, shape[1], shape[2]), dtype=np.float32)

        ns = np.arange(shape[1])
        one_cycle = 2 * np.pi * ns / shape[1]
        for k in range(shape[2]):
            t_k = k * one_cycle
            c[0, k, :] = np.cos(t_k)

        g = tf.constant(value=c, name=f"g", dtype=tf.float32)"""

        _g = tf.reshape(g, shape=(1, shape[1], shape[2], 1))
        tf.summary.image("g", _g, collections=['train'])
        tf.summary.histogram(f"g", _g, collections=['train'])

        n = tf.get_variable(f"n_{layer_idx}", shape=[shape[2], shape[3]],
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                            initializer=cur_layer.kernel_initializer,
                            regularizer=cur_layer.kernel_regularizer,
                            trainable=True)

        _n = tf.reshape(n, shape=(1, shape[2], shape[3], 1))
        tf.summary.image("n", _n, collections=['train'])
        tf.summary.histogram(f"n", _n, collections=['train'])

        tf.summary.histogram(f"n_{layer_idx}", n, collections=['train'])

        bias1 = None
        bias2 = None
        bias3 = None
        if cur_layer.using_bias():
            bias1 = tf.get_variable(f"bias1_{layer_idx}", shape=[shape[1]],  # W x H x C x r1
                                    initializer=cur_layer.bias_initializer,
                                    regularizer=cur_layer.bias_regularizer,
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                    trainable=True)

            tf.summary.histogram(f"bias1_{layer_idx}", bias1)

            bias2 = tf.get_variable(f"bias2_{layer_idx}", shape=[shape[2]],  # W x H x C x r2
                                    initializer=cur_layer.bias_initializer,
                                    regularizer=cur_layer.bias_regularizer,
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                    trainable=True)

            tf.summary.histogram(f"bias2_{layer_idx}", bias2)

            bias3 = tf.get_variable(f"bias3_{layer_idx}", shape=[shape[3]],  # W x H x C x n
                                    initializer=cur_layer.bias_initializer,
                                    regularizer=cur_layer.bias_regularizer,
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                    trainable=True)

            tf.summary.histogram(f"bias3_{layer_idx}", bias3)

    return Weights.PointwiseDot(c, g, n, bias1, bias2, bias3)


def CustomBottleneck(cur_layer, layer_idx):
    # W x H x C x N
    shape = cur_layer.get_shape()

    # Two partition variables.
    # pt1: Indicates % used for depthwise and standard convolution (concatenated at the end)
    # pt2: Indicates % used for factored pointwise convolution and standard pointwise convolution
    # The value % is how much is compressed (larger = more compressed)
    pt1 = 0.99
    pt2 = 0.99

    with tf.variable_scope("CustomBottleneck", reuse=tf.AUTO_REUSE):

        # Size of partition for first stage
        # Compressed across in-channels C
        s1 = int(shape[2] * pt1)  # Depthwise
        s2 = shape[2] - s1        # Standard convolution
        print(f"Depthwise: {s1}, Standard: {s2}")

        # Depthwise kernel
        kdw = tf.get_variable(f"kernel_depthwise_{layer_idx}", shape=[shape[0], shape[1], s1, 1],
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                              trainable=True)

        # Standard kernel
        k = tf.get_variable(f"kernel_{layer_idx}", shape=[shape[0], shape[1], s2, s2],
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                            trainable=True)

        # The size of auxilliary indices
        ranks = cur_layer.ranks
        assert len(ranks) == 3, "Must specified r0, r1, r2"

        k1 = Graph("conv_{}".format(layer_idx))

        # Size of partition for second stage
        # Compressed across out-channels (kernels) N
        s1 = int(shape[3] * pt2)  # Compressed
        s2 = shape[3] - s1        # Standard pointwise
        print(f"Factored PW: {s1}, Standard: {s2}")

        # Add the nodes w/ exposed indices
        k1.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
        k1.add_node("C", shape=[shape[2]], names=["C"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
        k1.add_node("N", shape=[s1], names=["N"],
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

        # Auxiliary indices
        # NOTE: Must specify shared at start
        k1.add_edge("WH", "G", name="r0", length=ranks[0], shared=True)
        k1.add_edge("C", "G", name="r1", length=ranks[1])
        k1.add_edge("N", "G", name="r2", length=ranks[2])

        # Compile/generate the tf.Variables and add to the set of weights
        k1.compile()
        k1.set_output_shape(["W", "H", "C", "N"])

        # Standard pointwise kernel
        k2 = tf.get_variable(f"k2_{layer_idx}", shape=[shape[0], shape[1], shape[2], s2],
                             collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                             initializer=cur_layer.kernel_initializer,
                             regularizer=cur_layer.kernel_regularizer,
                             trainable=True)

        # Some plots for Tensorboard
        c = k1.get_node("C")
        c_shape = c.get_shape().as_list()
        c = tf.reshape(c, shape=(1, c_shape[0], c_shape[1], 1))
        tf.summary.image(f"C_{layer_idx}", c, collections=['train'])

        n = k1.get_node("N")
        n_shape = n.get_shape().as_list()
        n = tf.reshape(n, shape=(1, n_shape[0], n_shape[1], 1))
        tf.summary.image(f"N_{layer_idx}", n, collections=['train'])

        g = k1.get_node("G")
        g_shape = g.get_shape().as_list()
        g = tf.reshape(g, shape=(g_shape[0], g_shape[1], g_shape[2], 1))
        tf.summary.image(f"G_{layer_idx}", g, collections=['train'])

        k2_shape = k2.get_shape().as_list()
        _k2 = tf.reshape(k2, shape=(s2, k2_shape[1], k2_shape[2], shape[0]))
        tf.summary.image(f"k2_{layer_idx}", _k2, collections=['train'])

        # Histograms
        tf.summary.histogram(f"k_{layer_idx}", k)
        tf.summary.histogram(f"kdw_{layer_idx}", kdw)

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x *N*
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.CustomBottleneck(k, kdw, k1, k2, bias)
