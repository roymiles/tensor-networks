""" Just call the layers normally, no core tensors, tensor contraction etc """
from Networks.network import Weights, INetwork
from Layers.impl.core import *
from base import *


class StandardNetwork(INetwork):
    def __init__(self, architecture):
        super(StandardNetwork, self).__init__()
        self.set_architecture(architecture)
        self._weights = None

    def run_layer(self, input, layer_idx, name, is_training=True, switch_idx=0):
        """ Pass input through a single layer
            Operation is dependant on the layer type

        :param input : input is a 4 dimensional feature map [B, W, H, C]
        :param layer_idx : Layer number
        :param name : For variable scoping
        :param is_training: bool, is training or testing mode
        :param switch_idx: Index for switch_list, controls the compression of the network
                       (default, just call first switch)

        """

        with tf.variable_scope(name):
            cur_layer = self.get_architecture().get_layer(layer_idx)

            if isinstance(cur_layer, ConvLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert isinstance(w, Weights.Convolution), "The layer weights don't match up with the layer type"

                if cur_layer.using_bias():
                    b = w.bias
                else:
                    b = None

                return cur_layer(input, kernel=w.kernel, bias=b)

            elif isinstance(cur_layer, DepthwiseConvLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert isinstance(w, Weights.DepthwiseConvolution), \
                    "The layer weights don't match up with the layer type"

                if cur_layer.using_bias():
                    b = w.bias
                else:
                    b = None

                return cur_layer(input, kernel=w.kernel, bias=b)

            elif isinstance(cur_layer, FullyConnectedLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert isinstance(w, Weights.FullyConnected), \
                    "The layer weights don't match up with the layer type"

                if cur_layer.using_bias():
                    b = w.bias
                else:
                    b = None

                return cur_layer(input, kernel=w.kernel, bias=b)

            elif isinstance(cur_layer, BatchNormalisationLayer):
                return cur_layer(input, is_training=is_training)

            elif isinstance(cur_layer, ReLU):
                act = INetwork.run_layer(layer=cur_layer, input=input)
                return act

            elif isinstance(cur_layer, MobileNetV2BottleNeck):

                w = self._weights.get_layer_weights(layer_idx)
                assert isinstance(w, Weights.Mobilenetv2Bottleneck), \
                    "The layer weights don't match up with the layer type"

                return cur_layer(input=input, expansion_kernel=w.expansion_kernel,
                                 expansion_bias=w.expansion_bias, depthwise_kernel=w.depthwise_kernel,
                                 depthwise_bias=w.depthwise_bias, projection_kernel=w.projection_kernel,
                                 projection_bias=w.projection_bias)

            else:
                # These layers are not overridden
                print(f"The following layer does not have a concrete implementation: {cur_layer}")
                return INetwork.run_layer(layer=cur_layer, input=input)
