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
    PointwiseDot = namedtuple('PointwiseDot', ["c", "g", "n", "bias1", "bias2", "bias3"])

    CustomBottleneck = namedtuple('CustomBottleneck', ["conv_kernel", "depthwise_kernel", "pointwise_kernel",
                                                       "factored_pointwise_kernel", "bias"])

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
        w = self._weights[layer_idx]
        return Weights.extract_tf_weights(w, switch)

    @staticmethod
    def extract_tf_weights(w, switch=1.0):
        d = w._asdict()

        # Convert each Graph member to tf.Variable by combining (contracting all the nodes)
        for name, value in d.items():
            if isinstance(value, Graph):
                d[name] = value.combine(switch)

        # Return the same namedtuple type, but with the updated(?) values
        return type(w)(*list(d.values()))

    def debug(self):
        for layer_idx, weight in self._weights.items():
            print("Layer {} -> {}".format(layer_idx, weight))
