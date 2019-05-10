import tensorflow as tf
import numpy as np
from layers import *


class TensorNetV1(tf.keras.Model):

    def __init__(self, architecture, conv_ranks, fc_ranks):
        """ Build an example tensor network
            The constructor effectively just initializes all the
            core tensors used in the tensor network

        Parameters
        ----------
        architecture : Contains the underlying architecture.
                       This is an object of type Architecture (see architectures.py)

        conv_ranks  : List of ranks (the same for every layer) [r1, r2, r3, r4]
        fc_ranks    : Same as conv_ranks, but 3 values [r1, r2, r3]

        Number of layers can be inferred from the network configuration
        """

        super(TensorNetV1, self).__init__(name='TensorNetV1')

        with tf.variable_scope("ExampleNetwork"):

            self.architecture = architecture
            self.conv_ranks = conv_ranks
            self.fc_ranks = fc_ranks
            self.num_layers = architecture.num_layers()

            # Wide compression suggests to use Gaussian initialization
            initializer = tf.contrib.layers.variance_scaling_initializer()

            # If they are not being used, just set the elements to -1 or None
            assert len(conv_ranks) == 4, "Must have 4 rank values for the convolution decomposition"
            assert len(fc_ranks) == 3, "Must have 3 rank values for the fully connected weight decomposition"

            # NOTE: These are dictionaries where the index is the layer_idx
            # Core tensors for convolutional layers
            self.t_wh = {}
            self.t_c = {}
            self.t_n = {}
            self.t_core = {}

            # Core tensors for fully connected layers
            self.t_in = {}
            self.t_out = {}
            self.t_fc = {}

            for layer_idx in range(self.num_layers):

                # Only need to initialize tensors for layers that have weights
                cur_layer = architecture.get_layer(layer_idx)
                if isinstance(cur_layer, ConvLayer):

                    shape = cur_layer.get_shape()

                    # W, H
                    self.t_wh[layer_idx] = tf.get_variable('t_wh_{}'.format(layer_idx),
                                                           shape=(shape[0], shape[1], conv_ranks[0]),
                                                           initializer=initializer)

                    # N
                    self.t_n[layer_idx] = tf.get_variable('t_n_{}'.format(layer_idx),
                                                          shape=(shape[3], conv_ranks[1], conv_ranks[2]),
                                                          initializer=initializer)

                    # C
                    self.t_c[layer_idx] = tf.get_variable('t_c_{}'.format(layer_idx),
                                                          shape=(shape[2], conv_ranks[2], conv_ranks[3]),
                                                          initializer=initializer)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    shape = cur_layer.get_shape()

                    self.t_in[layer_idx] = tf.get_variable('t_in_{}'.format(layer_idx),
                                                           shape=(shape[0], fc_ranks[0], fc_ranks[1]),
                                                           initializer=initializer)

                    self.t_out[layer_idx] = tf.get_variable('t_out_{}'.format(layer_idx),
                                                            shape=(shape[1], fc_ranks[1], fc_ranks[2]),
                                                            initializer=initializer)

            # Core tensors provides basis vectors
            # NOTE: This is the same across all layers
            # This high dimension core tensor is connected to all layer dim core tensors
            self.t_core = tf.get_variable('t_core', shape=(conv_ranks[0], conv_ranks[1], conv_ranks[3]),
                                          initializer=initializer)

            self.t_fc = tf.get_variable('t_fc', shape=(fc_ranks[0], fc_ranks[2]),
                                        initializer=initializer)

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
                # Spatial and core tensor
                # out [shape[0], shape[1], ranks[1], ranks[3]]
                c1 = tf.tensordot(self.t_wh[layer_idx], self.t_core, axes=[[2], [0]])

                # Channel tensors
                # out [shape[2], ranks[3], shape[3], ranks[1]]
                c2 = tf.tensordot(self.t_c[layer_idx], self.t_n[layer_idx], axes=[[1], [2]])

                # Final combine
                # out [shape[0], shape[1], shape[2], shape[3]]
                c3 = tf.tensordot(c1, c2, axes=[[2, 3], [3, 1]])

                print("----- {} -----".format(layer_idx))
                print("Input {}".format(input.get_shape()))
                print("Kernel for convolution = {}".format(c3.get_shape()))
                print("--------------")
                print("")

                # Call the function and return the result
                return cur_layer(input, c2)

            elif isinstance(cur_layer, FullyConnectedLayer):
                # Input and output channel core tensors
                # out [shape[2], ranks[5], shape[3], ranks[7]]
                c1 = tf.tensordot(self.t_in[layer_idx], self.t_out[layer_idx], axes=[[2], [1]])

                # Final combine
                # out [shape[2], shape[3]] - in, out
                c2 = tf.tensordot(c1, self.t_core, axes=[[2], [0]])

                print("----- {} -----".format(layer_idx))
                print("Input {}".format(input.get_shape()))
                print("Kernel for fc = {}".format(c2.get_shape()))
                print("--------------")
                print("")

                return cur_layer(input, c2)

            # For both pooling layers, there are no weights and so we
            # effectively just run their __call__ methods
            elif isinstance(cur_layer, AveragePoolingLayer):

                return cur_layer(input)

            elif isinstance(cur_layer, MaxPoolingLayer):

                return cur_layer(input)

            else:
                raise Exception("Invalid layer type")

    def call(self, input):
        """ Complete forward pass for the entire network """

        # Loop through all the layers
        net = input
        for n in range(self.num_layers):
            net = self.run_layer(net, layer_idx=n, name="layer_{}".format(n))

        # TODO: More, softmax etc
        return net

    def get_info(self):
        """ TODO: Print some helpful information about the network (pretty print) """
        print("Number of layers = {}".format(self.num_layers))

        # TODO: Why did this print not work?
        print("CONV Ranks ->".format(self.conv_ranks))
        print("FC Ranks ->".format(self.fc_ranks))

    def num_parameters(self):
        """ Get the total number of parameters (sum of all parameters in each core tensor) """

        n = 0
        for layer_idx in range(self.num_layers):

            if isinstance(self.architecture.get_layer(n), ConvLayer):

                for t in [self.t_wh[layer_idx], self.t_c[layer_idx], self.t_n[layer_idx]]:
                    n += np.prod(list(t.get_shape()))

                # Core tensor is not an array for each layer
                n += np.prod(list(self.t_core.get_shape()))

            elif isinstance(self.architecture.get_layer(n), FullyConnectedLayer):

                # Similarly for the fully connected layers
                for t in [self.t_in[layer_idx], self.t_out[layer_idx]]:
                    n += np.prod(list(t.get_shape()))

                n += np.prod(list(self.t_fc.get_shape()))

            else:
                raise Exception("Invalid layer type")

        return n
