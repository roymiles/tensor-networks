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
from Networks.network import INetwork, IWeights
from Networks.graph import Graph
import math


class Weights(IWeights):
    # NOTE: These are dictionaries where the index is the layer_idx

    # Container for the core tensors for the convolutional and fully connected layers
    # Each element is of type Graph
    conv_graph = {}
    fc_graph = {}

    # Biases (same dimensions for both fc and conv layers)
    bias = {}

    # Batch normalisation variables
    bn_mean = {}
    bn_variance = {}
    bn_scale = {}
    bn_offset = {}

    def __init__(self):
        pass

    def num_parameters(self):
        num_params = 0

        # Each element is a tf.Tensor
        for dict in [self.bias, self.bn_mean, self.bn_variance, self.bn_scale, self.bn_offset]:
            for layer_idx, variable in dict.items():
                num_params += tfvar_size(variable)

        # Each element is a Graph
        for dict in [self.conv_graph, self.fc_graph]:
            for layer_idx, graph in dict.items():
                num_params += graph.num_parameters()

        return num_params

    """ The weights are inferred from their argument name """
    def set_conv_layer_weights(self, layer_idx, conv_graph, bias):
        self.conv_graph[layer_idx] = conv_graph
        self.bias[layer_idx] = bias

    def set_fc_layer_weights(self, layer_idx, fc_graph, bias):
        self.fc_graph[layer_idx] = fc_graph
        self.bias[layer_idx] = bias

    def set_bn_layer_weights(self, layer_idx, mean, variance, scale, offset):
        self.bn_mean[layer_idx] = mean
        self.bn_variance[layer_idx] = variance
        self.bn_scale[layer_idx] = scale
        self.bn_offset[layer_idx] = offset

    def get_layer_weights(self, layer_idx):
        """ Get the weights for this given layer.
            If the weights e.g. conv_graph have the index layer_idx, then
            there are associated weights for this layer """
        if layer_idx in self.conv_graph:
            return {"__type__": LayerTypes.CONV, "graph": self.conv_graph[layer_idx], "bias": self.bias[layer_idx]}
        elif layer_idx in self.fc_graph:
            return {"__type__": LayerTypes.FC, "graph": self.fc_graph[layer_idx], "bias": self.bias[layer_idx]}
        elif layer_idx in self.bn_mean:  # Any of them will do
            return {"__type__": LayerTypes.BN, "mean": self.bn_mean[layer_idx], "variance": self.bn_variance[layer_idx],
                    "scale": self.bn_scale[layer_idx], "offset": self.bn_offset[layer_idx]}
        else:
            raise Exception("Unable to find a weight for this layer")


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

    def build(self, conv_ranks, fc_ranks, name, switch_list=[1]):
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

                    tensor_network = Graph("conv_{}".format(layer_idx))

                    # Add the nodes w/ exposed indices
                    tensor_network.add_node("WH", shape=[shape[0], shape[1]], names=["W", "H"])
                    tensor_network.add_node("C", shape=[shape[2]], names=["C"])
                    tensor_network.add_node("N", shape=[shape[3]], names=["N"])

                    # Auxilliary indices
                    tensor_network.add_edge("WH", "G", name="r0", length=ranks[0])
                    tensor_network.add_edge("C", "G", name="r1", length=ranks[1])
                    tensor_network.add_edge("N", "G", name="r2", length=ranks[2])

                    # Compile/generate the tf.Variables and add to the set of weights
                    tensor_network.compile()
                    self._weights.conv_graph[layer_idx] = tensor_network

                    if cur_layer.using_bias():
                        self._weights.bias[layer_idx] = tf.get_variable('bias_{}'.format(layer_idx),
                                                                        shape=shape[3],  # W x H x C x *N*
                                                                        initializer=initializer)

                elif isinstance(cur_layer, FullyConnectedLayer):

                    shape = cur_layer.get_shape()
                    ranks = fc_ranks[layer_idx]

                    tensor_network = Graph("fc_{}".format(layer_idx))

                    # Nodes..
                    tensor_network.add_node("I", shape=[shape[0]], names=["I"])
                    tensor_network.add_node("O", shape=[shape[1]], names=["O"])

                    # Auxilliary indices
                    tensor_network.add_edge("I", "G", name="r0", length=ranks[0])
                    tensor_network.add_edge("O", "G", name="r1", length=ranks[1])

                    # Compile the graph and add to the set of weights
                    tensor_network.compile()
                    self._weights.fc_graph[layer_idx] = tensor_network

                    if cur_layer.using_bias():
                        self._weights.bias[layer_idx] = tf.get_variable('bias_{}'.format(layer_idx),
                                                                        shape=shape[1],  # I x O
                                                                        initializer=initializer)

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

                        # Create the mean and variance weights
                        bn_mean.append(tf.get_variable(
                            'mean_l{}_s{}'.format(layer_idx, switch),
                            shape=num_features,
                            initializer=initializer)
                        )

                        bn_variance.append(tf.get_variable(
                            'variance_l{}_s{}'.format(layer_idx, switch),
                            shape=num_features,
                            initializer=initializer)
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
                                initializer=initializer)
                            )

                            bn_offset.append(tf.get_variable(
                                'offset_l{}_s{}'.format(layer_idx, switch),
                                shape=num_features,
                                initializer=initializer)
                            )

                    # Merge the array of parameters for different switches (of the same layer) into a single tensor
                    # e.g. to access first switch bn_mean[layer_idx][0]
                    self._weights.bn_mean[layer_idx] = tf.stack(bn_mean)
                    self._weights.bn_variance[layer_idx] = tf.stack(bn_variance)
                    self._weights.bn_scale[layer_idx] = tf.stack(bn_scale)
                    self._weights.bn_offset[layer_idx] = tf.stack(bn_offset)

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
                c = self._weights.conv_graph[layer_idx].combine(switch=switch)

                # TODO: This manual requirement for reshaping is bad. We know the names of edges,
                #       should be able to do this automatically like .reshape(["W", "H", "C", "N"])
                # Reshape to proper ordering
                s = tf.shape(c)
                c = tf.reshape(c, [s[2], s[3], s[1], s[0]])

                # Call the function and return the result
                return cur_layer(input, c, self._weights.bias[layer_idx])

            elif isinstance(cur_layer, FullyConnectedLayer):

                # assert 0 < switch <= 1, "Switch must be in the range (0, 1]"

                # Combine the core tensors
                c = self._weights.fc_graph[layer_idx].combine(switch=switch)

                # Reshape to proper ordering
                s = tf.shape(c)
                c = tf.reshape(c, [s[1], s[0]])

                if conf.is_debugging:
                    print("----- {} -----".format(layer_idx))
                    print("Input {}".format(input.get_shape()))
                    print("Kernel for fc = {}".format(c.get_shape()))
                    print("--------------")
                    print("")

                return cur_layer(input, c, self._weights.bias[layer_idx])

            elif isinstance(cur_layer, BatchNormalisationLayer):

                # Use the appropriate batch normalisation parameters for this given switch
                # Need to loop through all s in switch_list and check if ~= switch
                # switch_idx = -1
                # for i, s in enumerate(self._switch_list):
                #    if math.isclose(s, switch):
                #        switch_idx = i
                #        break
                # TODO: If not in list, do geometric mean of parameters between neighbouring switches
                # assert switch_idx != -1, "Unable to find appropriate switch from the switch list"

                return cur_layer(input, self._weights.bn_mean[layer_idx][switch_idx],
                                 self._weights.bn_variance[layer_idx][switch_idx],
                                 self._weights.bn_offset[layer_idx][switch_idx],
                                 self._weights.bn_offset[layer_idx][switch_idx])
            else:
                # These layers are not overridden
                return INetwork.run_layer(cur_layer, input=input)

    def __call__(self, input, switch_idx=0):
        """ Complete forward pass for the entire network

            :param
                input: The input to the network e.g. a batch of images
                switch_idx: Index for switch_list, controls the compression of the network
        """

        # assert 0 < switch <= 1, "Switch must be in the range (0, 1]"

        # Loop through all the layers
        net = input
        for n in range(self.get_num_layers()):
            net = self.run_layer(net, switch_idx=switch_idx, layer_idx=n, name="layer_{}".format(n))

        return net