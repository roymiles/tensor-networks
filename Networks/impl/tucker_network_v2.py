""" Use core tensors to parametrise the layers

    Affectively implementing Tucker decomposition
    g is a 3-way core tensor. C, N are matrix slices along each mode. W,H are combined
    This is the intent:


 y:::::::::::::::::::::::::s         //:::::::::::::::::::::::://         s:::::::::::::::::::::::::y
o                         o         /.                        ./         o                         o
o                         o         /.                        ./         o                         o
o                         o         /.                        ./         o                         o
o                         o         /.                        ./         o                         o
o                         o         /.                        ./         o                         o
o          W, H           o         /.            C           ./         o            N            o
o                         o         /.                        ./         o                         o
o                         o         /.                        ./         o                         o
o                         o         /.                        ./         o                         o
o                         o         /.                        ./         o                         o
o                         o         /.                        ./         o                         o
s`````````````````````````o         /-````````````````````````-/         o`````````````````````````s
:------------s-------------         .:-----------//-----------:.         -------------+------------:
             ::                                  ::                                   ::
             ::                                  ::                                   ::
             ::   R0                             ::   R1                              ::   R2
             ::                                  ::                                   ::
             ::                                  ::                                   ::
             ::                                  ::                                   ::
           ..s...................................//...................................s..
          `..-...................................//...................................-..`
                                                 ::
                                    .:-----------//-----------:.
                                    /-````````````````````````-/
                                    /.                        ./
                                    /.                        ./
                                    /.                        ./
                                    /.                        ./
                                    /.                        ./
                                    /.           g            ./
                                    /.                        ./
                                    /.                        ./
                                    /.                        ./
                                    /.                        ./
                                    /.                        ./
                                    //:::::::::::::::::::::::://

"""
from Layers.core import *
import config as conf
from base import *
from Networks.network import INetwork, IWeights
from Networks.graph import Graph


class Weights(IWeights):
    # NOTE: These are dictionaries where the index is the layer_idx

    # Container for the core tensors for the convolutional layers
    conv_graph = None

    # Biases (same dimensions for both fc and conv layers)
    bias = {}

    # Core tensors for fully connected layers
    fc_graph = None

    # Batch normalisation variables
    bn_mean = {}
    bn_variance = {}
    bn_scale = {}
    bn_offset = {}

    def __init__(self):
        pass

    def num_parameters(self):
        weight_list = []
        for dict in [self.conv_wh, self.conv_c, self.conv_n, self.fc_in, self.fc_out, self.bias, self.bn_mean,
                     self.bn_variance, self.bn_scale, self.bn_offset, self.conv_g, self.fc_g]:
            for layer_idx, variable in dict.items():
                weight_list.append(variable)

        return IWeights.num_parameters(weight_list)

    def from_layer(self, layer_idx):
        pass


class TuckerNet(INetwork):

    def __init__(self, architecture, conv_ranks, fc_ranks):
        """ Build an example tensor network
            The constructor effectively just initializes all the
            core tensors used in the tensor network

        Parameters
        ----------
        architecture : Contains the underlying architecture.
                       This is an object of type Architecture (see architectures.py)

        conv_ranks  : List of ranks (the same for every layer) [r1, r2, r3]

        Number of layers can be inferred from the network configuration
        """

        with tf.variable_scope("TensorNetwork"):

            self.architecture = architecture
            self.conv_ranks = conv_ranks
            self.num_layers = architecture.num_layers()

            # A container for all the core tensors, bn variables etc
            self._weights = Weights()

            for layer_idx in range(self.num_layers):

                # Only need to initialize tensors for layers that have weights
                cur_layer = architecture.get_layer(layer_idx)
                if isinstance(cur_layer, ConvLayer):

                    shape = cur_layer.get_shape()

                    self._weights.conv_graph = Graph

                    # Two exposed edges for WH dims
                    self._weights.conv_graph.add_edge("WH", "_D1_", shape[0], dummy_node=True)
                    self._weights.conv_graph.add_edge("WH", "_D2_", shape[1], dummy_node=True)
                    self._weights.conv_graph.add_edge("C", "_D3_", shape[2], dummy_node=True)
                    self._weights.conv_graph.add_edge("N", "_D4_", shape[3], dummy_node=True)

                    # Auxilliary indices
                    self._weights.conv_graph.add_edge("WH", "G", conv_ranks[0])
                    self._weights.conv_graph.add_edge("C", "G", conv_ranks[1])
                    self._weights.conv_graph.add_edge("N", "G", conv_ranks[2])

                    # Compile/generate the tf.Variables
                    self._weights.conv_graph.compile()

                    if cur_layer.using_bias():
                        self._weights.bias[layer_idx] = tf.get_variable('t_bias_{}'.format(layer_idx),
                                                                        shape=shape[3],  # W x H x C x *N*
                                                                        initializer=initializer)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    shape = cur_layer.get_shape()

                    self._weights.fc_in[layer_idx] = tf.get_variable('fc_in_{}'.format(layer_idx),
                                                                     shape=(shape[0], fc_ranks[0]),
                                                                     initializer=initializer)

                    self._weights.fc_out[layer_idx] = tf.get_variable('fc_out_{}'.format(layer_idx),
                                                                      shape=(shape[1], fc_ranks[1]),
                                                                      initializer=initializer)

                    self._weights.fc_g[layer_idx] = tf.get_variable('fc_g_{}'.format(layer_idx),
                                                                    shape=(fc_ranks[0], fc_ranks[1]),
                                                                    initializer=initializer)

                    if cur_layer.using_bias():
                        self._weights.bias[layer_idx] = tf.get_variable('t_bias_{}'.format(layer_idx),
                                                                        shape=shape[1],  # I x O
                                                                        initializer=initializer)

                elif isinstance(cur_layer, BatchNormalisationLayer):
                    # num_features is effectively the depth of the input feature map
                    num_features = cur_layer.get_num_features()

                    # Create the mean and variance weights
                    self._weights.bn_mean[layer_idx] = tf.get_variable('mean_{}'.format(layer_idx), shape=num_features,
                                                                       initializer=initializer)
                    self._weights.bn_variance[layer_idx] = tf.get_variable('variance_{}'.format(layer_idx), shape=num_features,
                                                                           initializer=initializer)

                    if cur_layer.is_affine():
                        # Scale (gamma) and offset (beta) parameters
                        self._weights.bn_scale[layer_idx] = tf.get_variable('scale_{}'.format(layer_idx),
                                                                            shape=num_features,
                                                                            initializer=initializer)
                        self._weights.bn_offset[layer_idx] = tf.get_variable('offset_{}'.format(layer_idx),
                                                                             shape=num_features,
                                                                             initializer=initializer)
                    else:
                        self._weights.bn_scale[layer_idx] = None
                        self._weights.bn_offset[layer_idx] = None

    def run_layer(self, input, layer_idx, name):
        """ Pass input through a single layer
            Operation is dependant on the layer type

        Parameters
        ----------
        input : input is a 4 dimensional feature map [B, W, H, C]
        layer_idx : Layer number
        name : For variable scoping
        """

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            # Merge core tensors using tensor contraction
            # Note: we take slices of the core tensors by layer idx
            # The spatial (w_h) core tensor must be merged with the core tensor

            cur_layer = self.architecture.get_layer(layer_idx)
            if isinstance(cur_layer, ConvLayer):

                # Combine the core tensors
                c = self._weights.conv_graph.combine()

                # Call the function and return the result
                return cur_layer(input, c, self._weights.bias[layer_idx])

            elif isinstance(cur_layer, FullyConnectedLayer):

                # Combine the core tensors
                c = self._weights.conv_graph.combine()

                if conf.is_debugging:
                    print("----- {} -----".format(layer_idx))
                    print("Input {}".format(input.get_shape()))
                    print("Kernel for fc = {}".format(c.get_shape()))
                    print("--------------")
                    print("")

                return cur_layer(input, c, self._weights.bias[layer_idx])

            elif isinstance(cur_layer, BatchNormalisationLayer):
                return cur_layer(input, self._weights.bn_mean[layer_idx], self._weights.bn_variance[layer_idx],
                                 self._weights.bn_offset[layer_idx], self._weights.bn_offset[layer_idx])
            else:
                # These layers are not overridden
                return INetwork.run_layer(cur_layer, input)

    def get_weights(self):
        return self._weights

    def set_weights(self, weights):
        self._weights = weights

    def __call__(self, input):
        """ Complete forward pass for the entire network """

        # Loop through all the layers
        net = input
        for n in range(self.num_layers):
            net = self.run_layer(net, layer_idx=n, name="layer_{}".format(n))

        return net

    def num_parameters(self):
        """ Get the total number of parameters (sum of all parameters in each core tensor) """
        return self._weights.num_parameters()
