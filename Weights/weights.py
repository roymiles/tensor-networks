"""
    Container for weights in a network
"""

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
    # Note: Named tuple is an immutable/constant dictionary with nicer lookup syntax :P
    Convolution = namedtuple('Convolution', ["kernel", "bias"])
    DepthwiseConvolution = namedtuple('DepthwiseConvolution', ["kernel", "bias"])
    FullyConnected = namedtuple('FullyConnected', ["kernel", "bias"])
    Mobilenetv2Bottleneck = namedtuple('Mobilenetv2Bottleneck', ["expansion_kernel", "depthwise_kernel",
                                                                 "projection_kernel"])
    PointwiseDot = namedtuple('PointwiseDot', ["c", "g", "n", "bias"])

    PartitionedDepthwiseSeparableLayer = namedtuple('PartitionedDepthwiseSeparableLayer',
                                                    ["conv_kernel", "depthwise_kernel", "pointwise_kernel",
                                                     "factored_pointwise_kernel", "bias"])

    DenseNetConvBlock = namedtuple('DenseNetConvBlock', ["pointwise_kernels", "conv_kernels"])

    # If we just want a (list?) of kernels
    JustKernels = namedtuple('JustKernels', ["kernel"])

    def __init__(self):
        pass

    def set_weights(self, layer_idx, tf_weights):
        self._weights[layer_idx] = tf_weights

    def get_layer_weights(self, layer_idx, switch=1.0):
        """
            Return the weights for a given layer
            Loops through all members in the weight namedtuple and combines them if of Graph type

            :return namedtuple w, where each member is a tf.Variable weight associated with this layer
                    e.g. w.kernel, w.bias
        """

        if layer_idx not in self._weights:
            # No weights for this layer
            return None

        w = self._weights[layer_idx]
        return Weights.extract_tf_weights(w, switch)

    @staticmethod
    def extract_tf_weights(w, switch=1.0):
        d = w._asdict()

        # Convert each Graph member to tf.Variable by combining (contracting all the nodes)
        for name, value in d.items():
            # If it is a list, go through each element
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, Graph):
                        d[name][i] = v.combine(switch)

            # If just single Graph element, convert it
            elif isinstance(value, Graph):
                d[name] = value.combine(switch)

            # If a tf.Variable, it is already in the correct format, so leave it.

        # Return the same namedtuple type, but with the updated(?) values
        return type(w)(*list(d.values()))

    def debug(self):
        for layer_idx, weight in self._weights.items():
            print("Layer {} -> {}".format(layer_idx, weight))
