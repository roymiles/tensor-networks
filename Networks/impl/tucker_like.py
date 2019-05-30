""" Use core tensors to parametrise the layers

    ** NOTE ** This is not actually tucker. Tucker uses W,H tensor as core (lower dimensional KxK spatial convolution)
                In this case, we have a separate core tensor h

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
from Layers.layer import LayerTypes
from Layers.impl.core import *
import config as conf
from base import *
from Networks.network import Weights, INetwork
from Networks.graph import Graph


class TuckerNet(INetwork):

    def __init__(self, architecture):
        """ Build an example tensor network
            The constructor effectively just initializes all the
            core tensors used in the tensor network

        Parameters
        ----------
        architecture : Contains the underlying architecture.
                       This is an object of type Architecture (see architectures.py)
        """

        super(TuckerNet, self).__init__()
        self.set_architecture(architecture)
        self._is_built = False
        self._switch_list = [1]

    def build(self, conv_ranks, fc_ranks, name, switch_list=[1.0]):
        """ Create/initialise the tf.Variable weights from ranks for ** ALL ** the layers
            This is not called if we are setting the weights manually

        Parameters
        ----------
        conv_ranks  : List of ranks (the same for every layer) {0:[r1, r2, r3], ... n:[...]}
        fc_ranks    : Same as above, but for fully connected layers {0:[r1, r2], ... n:[...]}
        name        : Tensorflow variable scope
        switch_list : List of all the switches with independent batch normalisation layers
                      By default we only use one switch (global batch normalisation parameters)

        The keys are the layer_idx

        Number of layers can be inferred from the network configuration
        """

        assert not self._is_built, "You cannot build the same Network more than once."

        with tf.variable_scope(name):

            # A container for all the core tensors, bn variables etc
            self._weights = Weights()

            # Need to keep track for calling the net
            self._switch_list = tf.constant(switch_list)

            for layer_idx in range(self.get_num_layers()):

                # Only need to initialize tensors for layers that have weights
                cur_layer = self.get_architecture().get_layer(layer_idx)
                if isinstance(cur_layer, ConvLayer):

                    shape = cur_layer.get_shape()
                    ranks = conv_ranks[layer_idx]

                    kernel = Graph("conv_{}".format(layer_idx))

                    # Add the nodes w/ exposed indices
                    kernel.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
                    kernel.add_node("C", shape=[shape[2]], names=["C"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
                    kernel.add_node("N", shape=[shape[3]], names=["N"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
                    #kernel.add_node("G", shape=[ranks[0], ranks[1], ranks[2]], names=["r0", "r1", "r2"],
                    #                collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])

                    # Auxiliary indices
                    kernel.add_edge("WH", "G", name="r0", length=ranks[0])
                    kernel.add_edge("C", "G", name="r1", length=ranks[1])
                    kernel.add_edge("N", "G", name="r2", length=ranks[2])

                    # Compile/generate the tf.Variables and add to the set of weights
                    kernel.compile()

                    if cur_layer.using_bias():
                        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x *N*
                        bias.add_node("B", shape=[shape[3]], names=["B"], initializer=tf.zeros_initializer(),
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])
                        bias.compile()

                        self._weights.set_conv_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_conv_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, DepthwiseConvLayer):

                    shape = cur_layer.get_shape()
                    ranks = conv_ranks[layer_idx]
                    kernel = Graph("dwconv_{}".format(layer_idx))

                    # Similar to standard convolution but with channel multiplier (M)
                    # instead of output channels dimension
                    kernel.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
                    kernel.add_node("C", shape=[shape[2]], names=["C"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
                    kernel.add_node("M", shape=[shape[3]], names=["M"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
                    #kernel.add_node("G", shape=[ranks[0], ranks[1], ranks[2]], names=["r0", "r1", "r2"],
                    #                collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])

                    kernel.add_edge("WH", "G", name="r0", length=ranks[0])
                    kernel.add_edge("C", "G", name="r1", length=ranks[1])
                    kernel.add_edge("M", "G", name="r2", length=ranks[2])
                    kernel.compile()

                    if cur_layer.using_bias():
                        bias = Graph("bias_{}".format(layer_idx))  # W x H x C x M
                        bias.add_node("B", shape=[shape[2] * shape[3]], names=["B"],
                                      initializer=tf.zeros_initializer(),
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"])  # Output channels is C x M
                        bias.compile()

                        self._weights.set_conv_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_conv_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    shape = cur_layer.get_shape()
                    ranks = fc_ranks[layer_idx]

                    kernel = Graph("fc_{}".format(layer_idx))

                    # Nodes..
                    kernel.add_node("I", shape=[shape[0]], names=["I"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
                    kernel.add_node("O", shape=[shape[1]], names=["O"],
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])
                    #kernel.add_node("G", shape=[ranks[0], ranks[1]], names=["r0", "r1"],
                    #                collections=[tf.GraphKeys.GLOBAL_VARIABLES, "weights"])

                    # Auxiliary indices
                    kernel.add_edge("I", "G1", name="r0", length=ranks[0])
                    kernel.add_edge("O", "G1", name="r1", length=ranks[1])

                    # Compile the graph and add to the set of weights
                    kernel.compile()

                    if cur_layer.using_bias():
                        bias = Graph("bias_{}".format(layer_idx))  # I x O
                        bias.add_node("B", shape=[shape[1]], names=["B"], initializer=tf.zeros_initializer(),
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, "bias"]).compile()

                        self._weights.set_fc_layer_weights(layer_idx, kernel, bias)
                    else:
                        # No bias term
                        self._weights.set_fc_layer_weights(layer_idx, kernel, None)

                elif isinstance(cur_layer, BatchNormalisationLayer):
                    # NOTE: We don't bother using tensor networks for the batch norm layers here
                    # num_features is effectively the depth of the input feature map
                    num_features = cur_layer.get_num_features()

                    # Independent parameters for each switch
                    bn_mean = []
                    bn_variance = []
                    bn_scale = []
                    bn_offset = []

                    for switch in switch_list:
                        # NOTE: Converting these to Graph is a pain because we have a stack of tf.Variables for each
                        # switch and they support placeholder indexing

                        # Create the mean and variance weights
                        bn_mean.append(tf.get_variable(
                            'mean_l{}_s{}'.format(layer_idx, switch),
                            shape=num_features,
                            initializer=tf.zeros_initializer())
                        )

                        bn_variance.append(tf.get_variable(
                            'variance_l{}_s{}'.format(layer_idx, switch),
                            shape=num_features,
                            initializer=tf.ones_initializer())
                        )

                        if cur_layer.is_affine():

                            # If set to None, this is equivalent to gamma=1, beta=0 (aka ignored)
                            bn_scale.append(None)
                            bn_offset.append(None)

                        else:
                            # Scale (gamma) and offset (beta) parameters
                            bn_scale.append(tf.get_variable(
                                'scale_l{}_s{}'.format(layer_idx, switch),
                                shape=num_features,
                                initializer=tf.ones_initializer())
                            )

                            bn_offset.append(tf.get_variable(
                                'offset_l{}_s{}'.format(layer_idx, switch),
                                shape=num_features,
                                initializer=tf.zeros_initializer())
                            )

                    # Merge the array of parameters for different switches (of the same layer) into a single tensor
                    # e.g. to access first switch bn_mean[layer_idx][0]
                    self._weights.set_bn_layer_weights(layer_idx, mean=tf.stack(bn_mean),
                                                       variance=tf.stack(bn_variance), scale=tf.stack(bn_scale),
                                                       offset=tf.stack(bn_offset))

        # All the tf Variables have been created
        self._is_built = True

    def run_layer(self, input, switch_idx, layer_idx, name):
        """ Pass input through a single layer
            Operation is dependant on the layer type

        Parameters
        ----------
        input : input is a 4 dimensional feature map [B, W, H, C]
        switch_idx: Index for switch_list
        layer_idx : Layer number
        name : For variable scoping
        """

        assert self._is_built, "The network must be built before you can run the layers."

        # Get the switch value
        switch = self._switch_list[switch_idx]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            # Merge core tensors using tensor contraction
            # Note: we take slices of the core tensors by layer idx
            # The spatial (w_h) core tensor must be merged with the core tensor

            cur_layer = self._architecture.get_layer(layer_idx)
            if isinstance(cur_layer, ConvLayer):

                # Convolutional and fully connected layers use switches
                # assert 0 < switch <= 1, "Switch must be in the range (0, 1]"

                # Combine the core tensors
                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.CONV, "The layer weights don't match up with the layer type"
                c = w["kernel"].combine(switch=switch, reshape=["W", "H", "C", "N"])

                if cur_layer.using_bias():
                    b = w["bias"].combine()
                else:
                    b = None

                # Call the function and return the result
                return cur_layer(input=input, kernel=c, bias=b)

            elif isinstance(cur_layer, FullyConnectedLayer):

                # assert 0 < switch <= 1, "Switch must be in the range (0, 1]"

                # Combine the core tensors
                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.FC, "The layer weights don't match up with the layer type"
                c = w["kernel"].combine(switch=switch, reshape=["I", "O"])

                if cur_layer.using_bias():
                    b = w["bias"].combine()
                else:
                    b = None

                return cur_layer(input=input, kernel=c, bias=b)

            elif isinstance(cur_layer, BatchNormalisationLayer):

                w = self._weights.get_layer_weights(layer_idx)
                assert w["__type__"] == LayerTypes.BN, "The layer weights don't match up with the layer type"

                return cur_layer(input=input, mean=w["mean"][switch_idx], variance=w["variance"][switch_idx],
                                 offset=w["offset"][switch_idx], scale=w["scale"][switch_idx])
            else:
                # These layers are not overridden
                return INetwork.run_layer(cur_layer, input=input)

    def __call__(self, input, switch_idx=0):
        """ Complete forward pass for the entire network

            :param
                input: The input to the network e.g. a batch of images
                switch_idx: Index for switch_list, controls the compression of the network
        """

        # Loop through all the layers
        net = input
        for n in range(self.get_num_layers()):
            net = self.run_layer(net, switch_idx=switch_idx, layer_idx=n, name="layer_{}".format(n))

        return net
