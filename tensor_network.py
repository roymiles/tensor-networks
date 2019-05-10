import tensorflow as tf
from tensor import Tensor
import numpy as np
from layers import *

class TensorNetV1(tf.keras.Model):

    def __init__(self, network_config, conv_ranks, fc_ranks):
        """ Build an example tensor network

        Parameters
        ----------
        network_config : Configuration for the architecture
                         The first index extracts the configuration details for a given layer e.g. layer_config = config[layer_idx]
                         A layer configuration is dependant on the layer type 
        
                         Size of convolutional kernels [[W_1, H_1, C_1, N_1], ... [W_N, H_N, C_N, N_N]]
                         Where N is the number of layers. This is required because each layer has different dimensions
                      
        conv_ranks  : List of ranks (the same for every layer) [r1, r2, r3, r4]
        fc_ranks    : Same as conv_ranks, but 3 values [r1, r2, r3]

        Number of layers can be infered from the network configuration
        """
        
        super(TensorNetV1, self).__init__(name='TensorNetV1')
        
        with tf.variable_scope("ExampleNetwork"):

            self.network_config = network_config
            self.conv_ranks = conv_ranks
            self.fc_ranks = fc_ranks
            self.num_layers = len(network_config)
            
            # Wide compression suggests to use Gaussian initialization
            initializer = tf.contrib.layers.variance_scaling_initializer()

            # If they are not being used, just set the elements to -1 or None
            assert len(conv_ranks) == 4, "Must have 4 rank values for the convolution decomposition"
            assert len(fc_ranks) == 3, "Must have 3 rank values for the fully connected weight decomposition"

            # Core tensors for convolutional layers
            self.t_wh = {}
            self.t_c = {}
            self.t_n = {}
            self.t_core = {}

            # Core tensors for fully connected layers
            self.t_in = {}
            self.t_out = {}
            self.t_fc = {}
            
            for n in range(self.num_layers):

                if isinstance(network_config[n], ConvLayer):
                    
                    # W, H
                    self.t_wh[n] = tf.get_variable('t_wh_{}'.format(n), shape=(shape[n][0], shape[n][1], conv_ranks[0]),
                                                   initializer=initializer)

                    # N
                    self.t_n[n] = tf.get_variable('t_n_{}'.format(n), shape=(shape[n][3], conv_ranks[1], conv_ranks[2]),
                                                  initializer=initializer)

                    # C
                    self.t_c[n] = tf.get_variable('t_c_{}'.format(n), shape=(shape[n][2], conv_ranks[2], conv_ranks[3]),
                                                  initializer=initializer)

                elif isinstance(network_config[n], FullyConnectedLayer):

                    self.t_in[n] = tf.get_variable('t_in_{}'.format(n), shape=(shape[n][0], fc_ranks[0], fc_ranks[1]),
                                                   initializer=initializer)

                    self.t_out[n] = tf.get_variable('t_out_{}'.format(n), shape=(shape[n][1], fc_ranks[1], fc_ranks[2]),
                                                    initializer=initializer)

                else:
                    raise Exception("Invalid layer type")

            # Core tensors provides basis vectors
            # NOTE: This is the same across all layers
            # This high dimension core tensor is connected to all layer dim core tensors
            self.t_core = tf.get_variable('t_core', shape=(conv_ranks[0], conv_ranks[1], conv_ranks[3]),
                                          initializer=initializer)
            
            self.t_fc = tf.get_variable('t_fc', shape=(fc_ranks[0], fc_ranks[2]),
                                        initializer=initializer)

    def forward(self, input, layer_idx):
        """ Pass input through a single layer (either a convolutional or fully connected layer)

        Parameters
        ----------
        input : input is a 4 dimensional image [B, W, H, C] (conv) or [B, N] (fc)
        layer_idx : Layer number
        """

        # Merge core tensors using tensor contraction
        # Note: we take slices of the core tensors by layer idx
        # The spatial (w_h) core tensor must be merged with the core tensor

        if isinstance(network_config[layer_idx], ConvLayer):
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
            
            return tf.nn.conv2d(input, c3, strides=[1, 1, 1, 1], padding="SAME")

        elif isinstance(network_config[layer_idx], FullyConnectedLayer):
            # Input and output channel core tensors
            # out [shape[2], ranks[5], shape[3], ranks[7]]
            c1 = tf.tensordot(self.t_in[layer_idx], self.t_out[layer_idx], axes=[[2], [1]])

            # Final combine
            # out [shape[2], shape[3]] - in, out
            c2 = tf.tensordot(c1, self.t_core, axes=[[2], [0]])

            # Flatten the input first
            input = tf.layers.flatten(input)
            
            print("----- {} -----".format(layer_idx))
            print("Input {}".format(input.get_shape()))
            print("Kernel for fc = {}".format(c2.get_shape()))
            print("--------------")
            print("")
            
            return tf.linalg.matmul(input, c2)
            
        elif isinstance(network_config[layer_idx], AveragePoolingLayer):
            # TODO: Finish
            
            
        else:
            raise Exception("Invalid layer type")

    def layer(self, input, layer_idx, name, use_relu=True):
        """ Perform a standard 2D convolution or fully connected layer, followed by batch norm and ReLU

        Parameters
        ----------
        input : input is a 4 dimensional image [B, W, H, C]
        layer_idx : Layer number
        name : For variable scoping
        """

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            net = self.forward(input, layer_idx)
            net = tf.layers.batch_normalization(net)
            
            if use_relu:
                net = tf.nn.relu(net)

            return net
            
    def call(self, input):
        """ Complete forward pass """
        
        # Loop through all the layers
        net = input
        for n in range(self.num_layers):
        
            use_relu = True
            if n == self.num_layers - 1:
                use_relu = False
                
            net = self.layer(net, layer_idx=n, name="layer_{}".format(n), use_relu=use_relu)
            
        # 

    def get_info(self):
        """ TODO: Print some helpful information about the network (pretty print) """
        print("Number of layers = {}".format(self.num_layers))

        for i, s in enumerate(self.shape):
            print("Kernel{} -> {}".format(i, s))

        # TODO: Why did this print not work??
        print("CONV Ranks ->".format(self.conv_ranks))
        print("FC Ranks ->".format(self.fc_ranks))

    def num_parameters(self):
        """ Get the total number of parameters (sum of all parameters in each core tensor) """

        n = 0
        for layer_idx in range(self.num_layers):

            if isinstance(self.network_config[n], ConvLayer):

                for t in [self.t_wh[layer_idx] , self.t_c[layer_idx] , self.t_n[layer_idx]]:
                    n += np.prod(list(t.get_shape()))

                # Core tensor is not an array for each layer
                n += np.prod(list(self.t_core.get_shape()))

            elif isinstance(self.network_config[n], FullyConnectedLayer):

                # Similarly for the fully connected layers
                for t in [self.t_in[layer_idx], self.t_out[layer_idx]]:
                    n += np.prod(list(t.get_shape()))

                n += np.prod(list(self.t_fc.get_shape()))

            else:
                raise Exception("Invalid layer type")

        return n
