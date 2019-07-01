import tensorflow as tf
from Weights.weights import Weights
from Networks.graph import Graph
import logging
from base import clamp


def convolution(cur_layer, layer_idx, name="Convolution"):
    """
        WH, C, and N each individually share an auxilliary index with a core tensor g
        This core tensor, g, is a 3-way tensor of shape [r1, r2, r3] (as specified in ranks)
    """
    with tf.variable_scope(name):
        shape = cur_layer.get_shape()

        # The size of auxilliary indices
        ranks = cur_layer.ranks
        assert len(ranks) == 3, "Must specified r0, r1, r2"

        kernel = Graph("graph_{}".format(layer_idx))

        # Add the nodes w/ exposed indices
        kernel.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
        kernel.add_node("C", shape=[shape[2]], names=["C"],
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
        kernel.add_node("N", shape=[shape[3]], names=["N"],
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

        # Auxiliary indices
        # NOTE: Must specify shared at start
        kernel.add_edge("WH", "G", name="r0", length=ranks[0])  # shared=true
        kernel.add_edge("C", "G", name="r1", length=ranks[1])
        kernel.add_edge("N", "G", name="r2", length=ranks[2])

        # Compile/generate the tf.Variables and add to the set of weights
        kernel.compile()
        kernel.set_output_shape(["W", "H", "C", "N"])

        # Some plots for Tensorboard
        kernel.create_summaries()

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x *N*
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.Convolution(kernel, bias)


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

        weight_decay = cur_layer.get_weight_decay()
        ranks = cur_layer.get_ranks()
        expansion = cur_layer.get_expansion()  # Expansion
        output_filters = cur_layer.get_filters()  # Number of output channels
        input_filters = cur_layer.get_in_channels()  # Number of input channels

        # H x W x k -> H x W x (tk)    1 x 1 x k x tk
        expansion_kernel = Graph("expansion_{}".format(layer_idx))

        # H x W x (tk) -> H/s x W/s x (tk)     3 x 3 x tk x 1   (last dim is depth multiplier)
        # Depthwise not using Graph

        # H/s x W/s x (tk) -> H/s x W/s x n     1 x 1 x tk x n
        projection_kernel = Graph("projection_{}".format(layer_idx))

        # Factorising the pointwise kernels
        expansion_kernel.add_node("WH", shape=[1, 1],
                                  names=["W", "H"],
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        expansion_kernel.add_node("C", shape=[input_filters],
                                  names=["C"],
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        expansion_kernel.add_node("N", shape=[expansion * input_filters],
                                  names=["N"],
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

        # Auxilliary ranks
        r1_c = 32  # int(ranks[0] * input_filters) + 16
        # r1_n =
        # int((input_filters * (expansion * input_filters) - (r1_c * input_filters)) / (expansion * input_filters))
        r1_n = 32  # int(ranks[1] * expansion * input_filters)

        # int(ranks[1] * expansion * input_filters)
        expansion_kernel.add_edge("WH", "G", name="r1", length=1, shared=True)
        expansion_kernel.add_edge("C", "G", name="r2", length=r1_c)
        expansion_kernel.add_edge("N", "G", name="r3", length=r1_n)
        expansion_kernel.compile()
        expansion_kernel.set_output_shape(["W", "H", "C", "N"])

        depthwise_kernel = tf.get_variable(f"depthwise_{layer_idx}", shape=[3, 3, input_filters * expansion, 1],
                                           collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                           initializer=tf.keras.initializers.glorot_normal(),
                                           trainable=True)

        # Projection layer
        # projection_kernel = tf.get_variable(f"projection_{layer_idx}",
        #                                    shape=[1, 1, input_filters*expansion, output_filters],
        #                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
        #                                    initializer=tf.keras.initializers.glorot_normal(),
        #                                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        #                                    trainable=True)

        # 1 x R*I x E*I
        # Flip it and use for projection
        """proj_g = tf.transpose(tf.squeeze(expansion_kernel.get_node("G"), name=f"projection_{layer_idx}_G"))
        proj_k = tf.get_variable(f"projection_{layer_idx}_K",
                                 shape=[input_filters * expansion, r_n],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                 initializer=tf.keras.initializers.glorot_normal(),
                                 regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 trainable=True)
        proj_m = tf.get_variable(f"projection_{layer_idx}_M",
                                 shape=[r_c, output_filters],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                 initializer=tf.keras.initializers.glorot_normal(),
                                 regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 trainable=True)

        p1 = tf.linalg.matmul(proj_k, proj_g)
        p2 = tf.linalg.matmul(p1, proj_m)
        projection_kernel = tf.reshape(p2, shape=[1, 1, expansion * input_filters, output_filters],
                                       name=f"projection_{layer_idx}")"""

        projection_kernel.add_node("WH", shape=[1, 1],
                                   names=["W", "H"],
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                   initializer=tf.keras.initializers.glorot_normal(),
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        projection_kernel.add_node("C", shape=[input_filters * expansion],
                                   names=["C"],
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                   initializer=tf.keras.initializers.glorot_normal(),
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        projection_kernel.add_node("N", shape=[output_filters],
                                   names=["N"],
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                   initializer=tf.keras.initializers.glorot_normal(),
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

        # r2_c = int(((expansion * input_filters) * output_filters - (r2_n * output_filters)) / input_filters)
        r2_c = 32  # int(ranks[2] * input_filters * expansion)
        r2_n = 32  # int(ranks[3] * output_filters) + 16

        projection_kernel.add_edge("WH", "G", name="r1", length=1, shared=True)
        projection_kernel.add_edge("C", "G", name="r2", length=r2_c)
        projection_kernel.add_edge("N", "G", name="r3", length=r2_n)
        # projection_kernel.add_edge("C", "G", name="r2", length=int(ranks[2] * output_filters))
        # projection_kernel.add_edge("N", "G", name="r3", length=int(ranks[3] * output_filters))
        projection_kernel.compile()
        projection_kernel.set_output_shape(["W", "H", "C", "N"])

        # Tensorflow summaries
        expansion_kernel.create_summaries()
        projection_kernel.create_summaries()

        return Weights.Mobilenetv2Bottleneck(expansion_kernel, depthwise_kernel, projection_kernel)


def pointwise_dot(cur_layer, layer_idx):
    shape = cur_layer.get_shape()

    with tf.variable_scope("PointwiseDot"):
        c = tf.get_variable(f"c_{layer_idx}", shape=[shape[0], shape[1]],
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                            initializer=cur_layer.kernel_initializer,
                            regularizer=cur_layer.kernel_regularizer,
                            trainable=True)

        tf.summary.histogram(f"c_{layer_idx}", c, collections=['train'])
        _c = tf.reshape(c, shape=(1, shape[0], shape[1], 1))
        tf.summary.image("c", _c, collections=['train'])
        tf.summary.histogram(f"c", _c, collections=['train'])

        # If sharing across layers (must have same shape)
        share_g = False
        if share_g:
            g_name = f"g"
        else:
            g_name = f"g_{layer_idx}"

        g = tf.get_variable(g_name, shape=[shape[1], shape[2]],
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                            initializer=cur_layer.kernel_initializer,
                            regularizer=cur_layer.kernel_regularizer,
                            trainable=True)

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

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x n
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

            tf.summary.histogram(f"bias_{layer_idx}", bias)

    return Weights.PointwiseDot(c, g, n, bias)


def custom_bottleneck(cur_layer, layer_idx):
    # W x H x C x N
    shape = cur_layer.get_shape()

    # Two partition variables: cur_layer.partitions[0]
    # 0: Indicates % used for depthwise and standard convolution (concatenated at the end)
    # 1: Indicates % used for factored pointwise convolution and standard pointwise convolution
    # Compression ratio. Smaller = More compression

    with tf.variable_scope("CustomBottleneck"):
        logging.info(f"CustomBottleneck Layer {layer_idx}")

        # Size of partition for first stage
        # Compressed across in-channels C

        # If the dimensions are large, we want to compress more (as this can mega inflate the number of parameters)
        # We do this by raising to the power of N, so domain kept to [0, 1]
        # Division by constant just makes it not blow up a lot (get reduced very very small)
        partitions = [cur_layer.partitions[0] ** (shape[2]/256),
                      cur_layer.partitions[1] ** (shape[3]/256)]

        conv_size = clamp(int(shape[2] * partitions[0]), 0, shape[2])  # Standard convolution
        depthwise_size = shape[2] - conv_size      # Depthwise
        logging.info(f"Depthwise: {depthwise_size}, Standard: {conv_size}")

        # Depthwise kernel
        depthwise_kernel = None
        if depthwise_size != 0:
            depthwise_kernel = tf.get_variable(f"depthwise_kernel_{layer_idx}",
                                               shape=[shape[0], shape[1], depthwise_size, 1],
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                               trainable=True)
            tf.summary.histogram(f"depthwise_kernel_{layer_idx}", depthwise_kernel, collections=['train'])

        # Standard kernel
        conv_kernel = None
        if conv_size != 0:
            conv_kernel = tf.get_variable(f"conv_kernel_{layer_idx}", shape=[shape[0], shape[1], conv_size, conv_size],
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                          trainable=True)
            tf.summary.histogram(f"conv_kernel_{layer_idx}", conv_kernel, collections=['train'])

        # Size of partition for second stage
        # Compressed across out-channels (kernels) N
        pointwise_size = clamp(int(shape[3] * partitions[1]), 0, shape[3])  # Standard pointwise
        factored_pointwise_size = shape[3] - pointwise_size       # Factored pointwise (compressed)
        logging.info(f"Factored PW: {factored_pointwise_size}, Standard: {pointwise_size}\n")

        factored_pointwise_kernel = None
        if factored_pointwise_size != 0:
            # The size of auxilliary indices
            ranks = cur_layer.ranks
            assert len(ranks) == 3, "Must specified r0, r1, r2"

            factored_pointwise_kernel = Graph("factored_pointwise_kernel_{}".format(layer_idx))

            # Add the nodes w/ exposed indices
            factored_pointwise_kernel.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
            factored_pointwise_kernel.add_node("C", shape=[shape[2]], names=["C"],
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
            factored_pointwise_kernel.add_node("N", shape=[factored_pointwise_size], names=["N"],
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

            # Auxiliary indices
            # NOTE: Must specify shared at start
            factored_pointwise_kernel.add_edge("WH", "G", name="r0", length=ranks[0], shared=True)
            factored_pointwise_kernel.add_edge("C", "G", name="r1", length=ranks[1])
            factored_pointwise_kernel.add_edge("N", "G", name="r2", length=ranks[2])

            # Compile/generate the tf.Variables and add to the set of weights
            factored_pointwise_kernel.compile()
            factored_pointwise_kernel.set_output_shape(["W", "H", "C", "N"])

            # Summaries / Histograms
            factored_pointwise_kernel.create_summaries()
            tf.summary.histogram(f"C_{layer_idx}", factored_pointwise_kernel.get_node("C"), collections=['train'])
            tf.summary.histogram(f"N_{layer_idx}", factored_pointwise_kernel.get_node("N"), collections=['train'])
            tf.summary.histogram(f"G_{layer_idx}", factored_pointwise_kernel.get_node("G"), collections=['train'])

        pointwise_kernel = None
        if pointwise_size != 0:
            # Standard pointwise kernel
            pointwise_kernel = tf.get_variable(f"pointwise_kernel_{layer_idx}", shape=[1, 1, shape[2], pointwise_size],
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                               trainable=True)
            tf.summary.histogram(f"pointwise_kernel_{layer_idx}", pointwise_kernel, collections=['train'])

        bias = None
        if cur_layer.using_bias():
            bias = tf.get_variable(f"bias_{layer_idx}", shape=[shape[3]],  # W x H x C x *N*
                                   initializer=cur_layer.bias_initializer,
                                   regularizer=cur_layer.bias_regularizer,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES],
                                   trainable=True)

        return Weights.CustomBottleneck(conv_kernel, depthwise_kernel, pointwise_kernel, factored_pointwise_kernel,
                                        bias)


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

            conv_kernel = Graph(f"conv_graph_{i}")

            # Add the nodes w/ exposed indices
            conv_kernel.add_node("WH", shape=[3, 3], names=["W", "H"],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
            conv_kernel.add_node("C", shape=[in_channels], names=["C"],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
            conv_kernel.add_node("N", shape=[cur_layer.growth_rate], names=["N"],
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

            # Auxiliary indices
            # NOTE: Must specify shared at start
            # Put more emphasis on spatial dimensions here. Transition layer is for depthwise.
            conv_kernel.add_edge("WH", "G", name="r0", length=16)
            conv_kernel.add_edge("C", "G", name="r1", length=int(in_channels/16))
            conv_kernel.add_edge("N", "G", name="r2", length=cur_layer.growth_rate/3)

            # Compile/generate the tf.Variables and add to the set of weights
            conv_kernel.compile()
            conv_kernel.set_output_shape(["W", "H", "C", "N"])

            # Was having some issues
            # Graph.debug(conv_kernel.get_graph(), f"debug_{i}")

            # Some plots for Tensorboard
            c = conv_kernel.get_node("C")
            c_shape = c.get_shape().as_list()
            c = tf.reshape(c, shape=(1, c_shape[0], c_shape[1], 1))
            tf.summary.image(f"C_{layer_idx}", c, collections=['train'])

            n = conv_kernel.get_node("N")
            n_shape = n.get_shape().as_list()
            n = tf.reshape(n, shape=(1, n_shape[0], n_shape[1], 1))
            tf.summary.image(f"N_{layer_idx}", n, collections=['train'])

            g = conv_kernel.get_node("G")
            g_shape = g.get_shape().as_list()
            g = tf.reshape(g, shape=(g_shape[0], g_shape[1], g_shape[2], 1))
            tf.summary.image(f"G_{layer_idx}", g, collections=['train'])

            # Add it to the list of kernels
            pointwise_kernels.append(pointwise_kernel)
            conv_kernels.append(conv_kernel)

            # Next layer is concatenation of input and output of previous layer
            in_channels += cur_layer.growth_rate

    # Reuse this interface but each element is a list for each subsequent bottleneck
    return Weights.DenseNetConvBlock(pointwise_kernels=pointwise_kernels, conv_kernels=conv_kernels)
