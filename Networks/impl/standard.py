""" Just call the layers normally, no core tensors, tensor contraction etc """
from Networks.network import Weights, INetwork
from Layers.layer import LayerTypes
from Layers.impl.core import *
import Layers.impl.keynet as KeyNetLayers
import tensorflow as tf
from base import *
from Networks.graph import Graph


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
        with tf.variable_scope(name):

            # All the weights of the network are stored in this container
            self._weights = Weights()

            # Initialize the standard convolutional and fully connected weights
            for layer_idx in range(self._num_layers):

                # Only need to initialize tensors for layers that have weights
                cur_layer = self.get_architecture().get_layer(layer_idx)
                if isinstance(cur_layer, ConvLayer):

                    shape = cur_layer.get_shape()

                    # A standard network just has a single high dimensional node
                    kernel = Graph("conv_{}".format(layer_idx))

                    kernel.add_node("WHCN", shape=[shape[0], shape[1], shape[2], shape[3]],
                                    names=["W", "H", "C", "N"], initializer=cur_layer.kernel_initializer,
                                    regularizer=cur_layer.kernel_regularizer,
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])

                    # Compile/generate the tf.Variables and add to the set of weights
                    kernel.compile()

                    if cur_layer.using_bias():

                        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x *N*
                        bias.add_node("B", shape=[shape[3]], names=["B"], initializer=cur_layer.bias_initializer,
                                      regularizer=cur_layer.bias_regularizer,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])
                        bias.compile()

                        self._weights.set_conv_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_conv_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, DepthwiseConvLayer):

                    # Very similar to standard convolution
                    shape = cur_layer.get_shape()
                    kernel = Graph("dwconv_{}".format(layer_idx))
                    kernel.add_node("WHCM", shape=[shape[0], shape[1], shape[2], shape[3]],
                                    names=["W", "H", "C", "M"], initializer=cur_layer.kernel_initializer,
                                    regularizer=cur_layer.kernel_regularizer,
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"]).compile()

                    if cur_layer.using_bias():

                        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x M
                        bias.add_node("B", shape=[shape[2] * shape[3]], names=["B"],
                                      initializer=cur_layer.bias_initializer, regularizer=cur_layer.bias_regularizer,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])
                        bias.compile()

                        self._weights.set_dw_conv_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_dw_conv_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    shape = cur_layer.get_shape()

                    # Create single node, compile the graph and then add to the set of weights
                    kernel = Graph("fc_{}".format(layer_idx))
                    kernel.add_node("IO", shape=[shape[0], shape[1]], names=["I", "O"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"]).compile()

                    if cur_layer.using_bias():

                        bias = Graph("bias_{}".format(layer_idx))  # I x O
                        bias.add_node("B", shape=[shape[1]], names=["B"], initializer=tf.zeros_initializer(),
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"]).compile()

                        self._weights.set_fc_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_fc_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, BatchNormalisationLayer):
                    # num_features is effectively the depth of the input feature map
                    num_features = cur_layer.get_num_features()

                    # Create the mean and variance weights
                    mean = Graph("mean_{}".format(layer_idx))
                    mean.add_node("M", shape=[num_features], names=["M"],
                                  initializer=cur_layer.moving_mean_initializer).compile()

                    variance = Graph("variance_{}".format(layer_idx))
                    variance.add_node("V", shape=[num_features], names=["V"],
                                      initializer=cur_layer.moving_variance_initializer).compile()

                    # When NOT affine
                    if not cur_layer.is_affine():
                        scale = None  # gamma
                        offset = None  # beta
                    else:
                        # Scale (gamma) and offset (beta) parameters
                        scale = Graph("scale_{}".format(layer_idx))
                        scale.add_node("S", shape=[num_features], names=["S"], initializer=cur_layer.gamma_initializer)
                        scale.compile()

                        offset = Graph("offset_{}".format(layer_idx))
                        offset.add_node("O", shape=[num_features], names=["O"], initializer=cur_layer.beta_initializer)
                        offset.compile()

                    self._weights.set_bn_layer_weights(layer_idx=layer_idx, mean=mean, variance=variance, scale=scale,
                                                       offset=offset)

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

            print(cur_layer)

            if isinstance(cur_layer, ConvLayer):

                w = self._weights.get_layer_weights(layer_idx)

                assert w["__type__"] == LayerTypes.CONV, "The layer weights don't match up with the layer type"

                c = w["kernel"].combine()
                print("c = {}".format(c))

                if cur_layer.using_bias():
                    b = w["bias"].combine()
                else:
                    b = None

                return cur_layer(input, kernel=c, bias=b)

            elif isinstance(cur_layer, DepthwiseConvLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.DW_CONV, "The layer weights don't match up with the layer type"

                c = w["kernel"].combine()
                b = w["bias"].combine()

                return cur_layer(input, kernel=c, bias=b)

            elif isinstance(cur_layer, FullyConnectedLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.FC, "The layer weights don't match up with the layer type"

                c = w["kernel"].combine()
                b = w["bias"].combine()

                return cur_layer(input, kernel=c, bias=b)

            elif isinstance(cur_layer, BatchNormalisationLayer):

                w = self._weights.get_layer_weights(layer_idx)

                mean = w["mean"]
                if isinstance(mean, Graph):
                    mean = mean.combine()

                variance = w["variance"]
                if isinstance(variance, Graph):
                    variance = variance.combine()

                scale = w["scale"]
                if isinstance(scale, Graph):
                    scale = scale.combine()

                offset = w["offset"]
                if isinstance(offset, Graph):
                    offset = offset.combine()

                return cur_layer(input, mean, variance, scale, offset)
            elif isinstance(cur_layer, ReLU):
                act = INetwork.run_layer(layer=cur_layer, input=input, **kwargs)
                return act
            else:
                # These layers are not overridden
                print("Woah, are you sure you should be here with: {}?".format(cur_layer))
                return INetwork.run_layer(layer=cur_layer, input=input, **kwargs)

    def __call__(self, input, switch_idx=0, dense_connect=True):
        """ Complete forward pass for the entire network

            :param input: The input to the network e.g. a batch of images
            :param switch_idx: Index for switch_list, controls the compression of the network
                               (default, just call first switch)
            :param dense_connect: Densely connect all layers (as per DenseNet)
        """

        # Loop through all the layers
        if dense_connect:

            # Densely connected network - combined through concatenation
            net = [input]  # Store all layer outputs
            for n in range(self.get_num_layers()):
                print("net: {}".format(net))
                inp = tf.concat(net, axis=3)
                print("inp: {}".format(inp))
                out = self.run_layer(input=inp, layer_idx=n, name="layer_{}".format(n))
                net.append(out)

            return net

        else:
            net = input
            for n in range(self.get_num_layers()):
                net = self.run_layer(input=net, layer_idx=n, name="layer_{}".format(n))

            return net
