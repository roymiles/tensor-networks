"""
    These utility functions return the tf.Variables used for a given layer
    They can be used in different tensor networks.

    This is because most tensor network implementations use common structure for some layers, such as batch norm

    None of these layers should use Graph, as the tensors are not decomposed. Instead tf.get_variable is used directly.
"""

import tensorflow as tf
from Networks.graph import Graph
from collections import namedtuple


class Weights:
    """
        All networks use this weight data structure to store and query the weights values
        The weight values are either stored as tensor networks or tf.Variables
    """

    # Stored as key:value pair, where key is a layer_idx
    _weights = {}

    # All the types of weights
    Convolution = namedtuple('Convolution', ["kernel", "bias"])
    DepthwiseConvolution = namedtuple('DepthwiseConvolution', ["kernel", "bias"])
    FullyConnected = namedtuple('FullyConnected', ["kernel", "bias"])
    Mobilenetv2Bottleneck = namedtuple('Mobilenetv2Bottleneck', ["expansion_kernel", "expansion_bias",
                                                                 "depthwise_kernel", "depthwise_bias",
                                                                 "projection_kernel", "projection_bias"])

    def __init__(self):
        pass

    def set_weights(self, layer_idx, tf_weights):
        self._weights[layer_idx] = tf_weights

    def get_layer_weights(self, layer_idx):
        """
            Return the weights for a given layer
            Loops through all members in the weight namedtuple and combines them if of Graph type
        """
        w = self._weights[layer_idx]
        # for name, value in w._asdict().iteritems():
        return w

    def debug(self):
        for layer_idx, weight in self._weights.items():
            print("Layer {} -> {}".format(layer_idx, weight))


class CreateWeights:

    class Core:
        """
            Standard weights
        """
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

            tf.summary.histogram(f"dwconv_{layer_idx}", kernel)

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

                tf.summary.histogram(f"dwconv_bias_{layer_idx}", bias)

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

    class Sandbox:
        @staticmethod
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

            bias = None
            if cur_layer.using_bias():
                bias = Graph("bias_{}".format(layer_idx))  # W x H x C x *N*
                bias.add_node("B", shape=[shape[3]], names=["B"], initializer=tf.zeros_initializer(),
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])
                bias.compile()

            return Weights.Convolution(kernel, bias)

        @staticmethod
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

        @staticmethod
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