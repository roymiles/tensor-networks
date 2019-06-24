""" Define interfaces for a network and the weights inside the network """
from Layers.impl.core import *
from Architectures.architectures import IArchitecture
from Weights.weights import Weights
import tensorflow as tf


class Network:

    def __init__(self, architecture):
        assert isinstance(architecture, IArchitecture), "architecture argument must be of type IArchitecture"
        self._architecture = architecture
        self._num_layers = architecture.num_layers()

        self._weights = None

    def set_weights(self, weights):

        assert isinstance(weights, Weights), "weights must be of type IWeights"
        self._weights = weights

    def get_weights(self):
        """

        :return: IWeights: Return all the weights for the network
        """
        return self._weights

    def get_num_layers(self):
        """
        NOTE: There is no setter, this value is inferred when set_architecture is called

        :return: The number of layers in the architecture
        """
        return self._num_layers

    def get_architecture(self):
        """ Return the underlying architecture of this network, will be of type IArchitecture """
        return self._architecture

    def num_parameters(self):
        """ Get the total number of parameters in the network
            For example, in a Tucker network this will be the sum of all parameters in each core tensor """
        return self._weights.num_parameters()

    def build(self, name="weights"):
        """
            Build the tf.Variable weights used by the network

            :param name: Variable scope e.g. "StandardNetwork1"
        """
        with tf.variable_scope(name):

            # All the weights of the network are stored in this container
            self._weights = Weights()

            # Initialize the standard convolutional and fully connected weights
            for layer_idx in range(self._num_layers):

                # Only need to initialize tensors for layers that have weights
                cur_layer = self.get_architecture().get_layer(layer_idx)

                # If the current layer has a create_weights function, call it and add the weights
                # to the set of weights
                create_op = getattr(cur_layer, "create_weights", None)
                if callable(create_op):
                    tf_weights = create_op()(cur_layer, layer_idx)
                    self._weights.set_weights(layer_idx, tf_weights)

    def run_layer(self, x, layer_idx, name, is_training=True, switch_idx=0, switch=1.0):
        """ Pass input through a single layer
            Operation is dependant on the layer type

        :param x : input is a 4 dimensional feature map [B, W, H, C]
        :param layer_idx : Layer number
        :param name : For variable scoping
        :param is_training: bool, is training or testing mode
        :param switch_idx: Associated switch index from the switch list
        :param switch: Switch in range 0-1
                       (default, use full network)

        """

        with tf.variable_scope(name):
            cur_layer = self.get_architecture().get_layer(layer_idx)

            if isinstance(cur_layer, ConvLayer):

                w = self._weights.get_layer_weights(layer_idx)

                assert isinstance(w, Weights.Convolution), \
                    "The layer weights don't match up with the layer type"

                return cur_layer(x, kernel=w.kernel, bias=w.bias)

            elif isinstance(cur_layer, CustomBottleneck):
                w = self._weights.get_layer_weights(layer_idx)

                assert isinstance(w, Weights.CustomBottleneck), \
                    "The layer weights don't match up with the layer type"

                return cur_layer(x, conv_kernel=w.conv_kernel, depthwise_kernel=w.depthwise_kernel,
                                 pointwise_kernel=w.pointwise_kernel,
                                 factored_pointwise_kernel=w.factored_pointwise_kernel,
                                 is_training=is_training, bias=w.bias)

            elif isinstance(cur_layer, DepthwiseConvLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert isinstance(w, Weights.DepthwiseConvolution), \
                    "The layer weights don't match up with the layer type"

                return cur_layer(x, kernel=w.kernel, bias=w.bias)

            elif isinstance(cur_layer, FullyConnectedLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert isinstance(w, Weights.FullyConnected), \
                    "The layer weights don't match up with the layer type"

                return cur_layer(x, kernel=w.kernel, bias=w.bias)

            elif isinstance(cur_layer, BatchNormalisationLayer):
                return cur_layer(x, is_training=is_training, switch_idx=switch_idx)

            elif isinstance(cur_layer, ReLU):  # issubclass(cur_layer, NonLinearityLayer):
                """ Any nonlinearity, ReLU, HSwitch etc have the same interface """
                act = cur_layer(x)
                return act

            elif isinstance(cur_layer, MobileNetV2BottleNeck):

                w = self._weights.get_layer_weights(layer_idx)
                assert isinstance(w, Weights.Mobilenetv2Bottleneck), \
                    "The layer weights don't match up with the layer type"

                return cur_layer(x, expansion_kernel=w.expansion_kernel, depthwise_kernel=w.depthwise_kernel,
                                 projection_kernel=w.projection_kernel)

            elif isinstance(cur_layer, PointwiseDot):

                w = self._weights.get_layer_weights(layer_idx)
                assert isinstance(w, Weights.PointwiseDot), \
                    "The layer weights don't match up with the layer type"

                return cur_layer(x, c=w.c, g=w.g, n=w.n, bias1=w.bias1, bias2=w.bias2, bias3=w.bias3,
                                 is_training=is_training)

            else:
                print(f"The following layer does not have a concrete implementation: {cur_layer}")
                return cur_layer(x)

    def __call__(self, x, is_training, switch_idx=0, switch=1.0):
        """ Complete forward pass for the entire network

            :param x: The input to the network e.g. a batch of images
            :param switch_idx: Associated switch index from the switch list
            :param switch: Current switch in range 0, 1
            :param is_training: bool, is training or testing mode
        """

        tf.summary.image("input_data", x, collections=['train', 'test'])

        with tf.variable_scope("network"):
            # Loop through all the layers
            net = x
            for n in range(self.get_num_layers()):
                net = self.run_layer(net, layer_idx=n, name=f"layer_{n}",
                                     is_training=is_training, switch_idx=switch_idx, switch=switch)

            return net
