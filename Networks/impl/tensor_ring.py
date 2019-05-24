""" Use core tensors to parametrise the layers

    TODO: Check if this is actually tensor ring net

    This is the intent:


    **** SIGNIFICANTLY OUTDATED *** SEE STANDARD/TENSOR_LIKE


     :/----------------------s        :+----------------------s        -+----------------------s
    :-                      o        :-                      o        --                      o
    :-                      o        :-                      o        --                      o
    :-                      o        :-                      o        --                      o
    :-                      o        :-                      o        --                      o
  ``/-                      o````````/-                      o````````:-                      o```
--../-         W, H         o......../-          C           o......../-          N           o...:
+   :-                      o        :-                      o        --                      o   .:
+   :-                      o        :-                      o        --                      o    +
o   :-                      o        :-                      o        --                      o    o
o   :-                      o        :-                      o        --                      o    o
o   ::......................o        ::......................o        -/......................o    o
o   `.``````````````````````.        `.``````````````````````.        `.``````````````````````.    o
o                                                                                                  o
o                                                                                                  o
o                                                                                                  o
o                                    `.``````````````````````.                                     o
o                                    ::......................o                                     o
o                                    :-                      o                                     o
o                                    :-                      o                                     o
+                                    :-                      o                                    `+
+`                                   :-                      o                                    :-
`---................................./-                      o..................................--.
   ``````````````````````````````````/-          g           o``````````````````````````````````
                                     :-                      o
                                     :-                      o
                                     :-                      o
                                     :-                      o
                                     :+----------------------s

"""
from Layers.layer import *
import config as conf
from base import *
from Networks.network import Weights, INetwork


class TensorRingNet(INetwork):

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

        with tf.variable_scope("TensorNetwork"):

            self.architecture = architecture
            self.conv_ranks = conv_ranks
            self.fc_ranks = fc_ranks
            self.num_layers = architecture.num_layers()

            # If they are not being used, just set the elements to -1 or None
            assert len(conv_ranks) == 4, "Must have 4 rank values for the convolution decomposition"
            assert len(fc_ranks) == 3, "Must have 3 rank values for the fully connected weight decomposition"

            # A container for all the core tensors, bn variables etc
            self._weights = Weights()

            for layer_idx in range(self.num_layers):

                # Only need to initialize tensors for layers that have weights
                cur_layer = architecture.get_layer(layer_idx)
                if isinstance(cur_layer, ConvLayer):

                    shape = cur_layer.get_shape()

                    # W, H
                    self._weights.conv_wh[layer_idx] = tf.get_variable('t_wh_{}'.format(layer_idx),
                                                                       shape=(shape[0], shape[1], conv_ranks[0]),
                                                                       initializer=initializer)

                    # N
                    self._weights.conv_n[layer_idx] = tf.get_variable('t_n_{}'.format(layer_idx),
                                                                      shape=(shape[3], conv_ranks[1], conv_ranks[2]),
                                                                      initializer=initializer)

                    # C
                    self._weights.conv_c[layer_idx] = tf.get_variable('t_c_{}'.format(layer_idx),
                                                                      shape=(shape[2], conv_ranks[2], conv_ranks[3]),
                                                                      initializer=initializer)

                    if cur_layer.using_bias():
                        self._weights.bias[layer_idx] = tf.get_variable('t_bias_{}'.format(layer_idx),
                                                                        shape=shape[3],  # W x H x C x N
                                                                        initializer=initializer)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    shape = cur_layer.get_shape()

                    self._weights.fc_in[layer_idx] = tf.get_variable('t_in_{}'.format(layer_idx),
                                                                     shape=(shape[0], fc_ranks[0], fc_ranks[1]),
                                                                     initializer=initializer)

                    self._weights.fc_out[layer_idx] = tf.get_variable('t_out_{}'.format(layer_idx),
                                                                      shape=(shape[1], fc_ranks[1], fc_ranks[2]),
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

            # Core tensors provides basis vectors
            # NOTE: This is the same across all layers
            # This high dimension core tensor is connected to all layer dim core tensors
            self._weights.conv_core = tf.get_variable('t_core', shape=(conv_ranks[0], conv_ranks[1], conv_ranks[3]),
                                                      initializer=initializer)

            self._weights.fc_core = tf.get_variable('t_fc', shape=(fc_ranks[0], fc_ranks[2]),
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
                c1 = tf.tensordot(self._weights.conv_wh[layer_idx], self._weights.conv_core, axes=[[2], [0]])

                # Channel tensors
                # out [shape[2], ranks[3], shape[3], ranks[1]]
                c2 = tf.tensordot(self._weights.conv_c[layer_idx], self._weights.conv_n[layer_idx], axes=[[1], [2]])

                # Final combine
                # out [shape[0], shape[1], shape[2], shape[3]]
                c3 = tf.tensordot(c1, c2, axes=[[2, 3], [3, 1]])

                if conf.is_debugging:
                    print("----- {} -----".format(layer_idx))
                    print("Input {}".format(input.get_shape()))
                    print("Kernel for convolution = {}".format(c3.get_shape()))
                    print("--------------")
                    print("")

                # Call the function and return the result
                return cur_layer(input, c3, self._weights.bias[layer_idx])

            elif isinstance(cur_layer, FullyConnectedLayer):
                # Input and output channel core tensors
                # out [shape[0], ranks[0], shape[1], ranks[2]]
                c1 = tf.tensordot(self._weights.fc_in[layer_idx], self._weights.fc_out[layer_idx], axes=[[2], [1]])

                # Final combine
                # out [shape[0], shape[1]] - in, out
                c2 = tf.tensordot(c1, self._weights.fc_core, axes=[[1, 3], [0, 1]])

                if conf.is_debugging:
                    print("----- {} -----".format(layer_idx))
                    print("Input {}".format(input.get_shape()))
                    print("Kernel for fc = {}".format(c2.get_shape()))
                    print("--------------")
                    print("")

                return cur_layer(input, c2, self.t_bias[layer_idx])

            elif isinstance(cur_layer, BatchNormalisationLayer):
                return cur_layer(input, self._weights.bn_mean[layer_idx], self._weights.bn_variance[layer_idx],
                                 self._weights.bn_offset[layer_idx], self._weights.bn_offset[layer_idx])
            else:
                # These layers are not overridden
                return INetwork.run_layer(cur_layer, input)

    def __call__(self, input):
        """ Complete forward pass for the entire network """

        # Loop through all the layers
        net = input
        for n in range(self.num_layers):
            net = self.run_layer(net, layer_idx=n, name="layer_{}".format(n))

        return net