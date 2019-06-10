""" Just call the layers normally, no core tensors, tensor contraction etc """
from Networks.network import Weights, INetwork
from Layers.layer import LayerTypes
from Layers.impl.core import *
from base import *
from Networks.graph import Graph
import Networks.Layers.core as nl


class StandardNetwork(INetwork):
    def __init__(self, architecture):
        super(StandardNetwork, self).__init__()
        self.set_architecture(architecture)
        self._weights = None

    def build(self, name):
        """
            Build the tf.Variable weights used by the network

            :param name: Variable scope e.g. StandardNetwork1
        """
        # with tf.variable_scope(name):

        # All the weights of the network are stored in this container
        self._weights = Weights()

        # Initialize the standard convolutional and fully connected weights
        for layer_idx in range(self._num_layers):

            # Only need to initialize tensors for layers that have weights
            cur_layer = self.get_architecture().get_layer(layer_idx)
            if isinstance(cur_layer, ConvLayer):
                tf_weights = nl.convolution(cur_layer, layer_idx)
                self._weights.set_conv_layer_weights(layer_idx, **tf_weights)

            elif isinstance(cur_layer, DepthwiseConvLayer):
                tf_weights = nl.depthwise_convolution(cur_layer, layer_idx)
                self._weights.set_dw_conv_layer_weights(layer_idx, **tf_weights)

            elif isinstance(cur_layer, FullyConnectedLayer):
                tf_weights = nl.fully_connected(cur_layer, layer_idx)
                self._weights.set_fc_layer_weights(layer_idx, **tf_weights)

            elif isinstance(cur_layer, BatchNormalisationLayer):
                tf_weights = nl.batch_normalisation(cur_layer, layer_idx)
                self._weights.set_bn_layer_weights(layer_idx, **tf_weights)

            elif isinstance(cur_layer, MobileNetV2BottleNeck):
                tf_weights = nl.mobilenetv2_bottleneck(cur_layer, layer_idx)
                self._weights.set_mobilenetv2_bottleneck_layer_weights(layer_idx, **tf_weights)

    def run_layer(self, input, layer_idx, name, **kwargs):
        """ Pass input through a single layer
            Operation is dependant on the layer type

        :param input : input is a 4 dimensional feature map [B, W, H, C]
        :param layer_idx : Layer number
        :param name : For variable scoping

        Additional parameters are named in kwargs
        """

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            cur_layer = self.get_architecture().get_layer(layer_idx)

            if isinstance(cur_layer, ConvLayer):

                w = self._weights.get_layer_weights(layer_idx)

                assert w["__type__"] == LayerTypes.CONV, "The layer weights don't match up with the layer type"

                c = w["kernel"]

                if cur_layer.using_bias():
                    b = w["bias"]
                else:
                    b = None

                return cur_layer(input, kernel=c, bias=b)

            elif isinstance(cur_layer, DepthwiseConvLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.DW_CONV, "The layer weights don't match up with the layer type"

                c = w["kernel"]
                b = w["bias"]

                return cur_layer(input, kernel=c, bias=b)

            elif isinstance(cur_layer, FullyConnectedLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.FC, "The layer weights don't match up with the layer type"

                c = w["kernel"]
                b = w["bias"]

                return cur_layer(input, kernel=c, bias=b)

            elif isinstance(cur_layer, BatchNormalisationLayer):

                w = self._weights.get_layer_weights(layer_idx)

                mean = w["mean"]
                variance = w["variance"]
                scale = w["scale"]
                offset = w["offset"]

                return cur_layer(input, mean, variance, scale, offset)

            elif isinstance(cur_layer, ReLU):
                act = INetwork.run_layer(layer=cur_layer, input=input)
                return act

            elif isinstance(cur_layer, MobileNetV2BottleNeck):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.MOBILENETV2_BOTTLENECK, \
                    "The layer weights don't match up with the layer type"

                expansion_kernel = w["expansion_kernel"]
                expansion_bias = w["expansion_bias"]
                depthwise_kernel = w["depthwise_kernel"]
                depthwise_bias = w["depthwise_bias"]
                projection_kernel = w["projection_kernel"]
                projection_bias = w["projection_bias"]

                return cur_layer(input=input, expansion_kernel=expansion_kernel,
                                 expansion_bias=expansion_bias, depthwise_kernel=depthwise_kernel,
                                 depthwise_bias=depthwise_bias, projection_kernel=projection_kernel,
                                 projection_bias=projection_bias)

            else:
                # These layers are not overridden
                print("The following layer does not have a concrete implementation: {}?".format(cur_layer))
                return INetwork.run_layer(layer=cur_layer, input=input, **kwargs)

    def __call__(self, input, switch_idx=0):
        """ Complete forward pass for the entire network

            :param input: The input to the network e.g. a batch of images
            :param switch_idx: Index for switch_list, controls the compression of the network
                               (default, just call first switch)
        """

        # Loop through all the layers
        net = input
        for n in range(self.get_num_layers()):
            net = self.run_layer(input=net, layer_idx=n, name="layer_{}".format(n))

        return net
