""" Convert a network between types """

from Networks.impl.standard import StandardNetwork
from Networks.impl.sandbox import TuckerNet
import VBMF
import numpy as np
import tensorflow as tf
from Layers.impl.core import ConvLayer, FullyConnectedLayer

import tensorly as tl
# If use tensorflow, it gets confusing with eager execution in parts
tl.set_backend('numpy')


def estimate_ranks(weights):
    """ Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """

    raise Exception("TODO: Does not quite work")

    print(weights.numpy().shape)
    unfold_0 = tl.base.unfold(weights, 0).numpy()
    print(unfold_0.shape)
    unfold_1 = tl.base.unfold(weights, 1).numpy()
    print(unfold_1.shape)

    a, diag_0, b, c = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    print("here {} {} {} {} there".format(a, diag_0, b, c))
    ranks = [diag_0.shape[0], diag_1.shape[1]]

    a = tf.constant(np.random.randint(100, size=(3, 3, 32, 64)))
    b = tl.base.unfold(a, 2).numpy()
    c = tl.base.unfold(a, 3).numpy()
    print(a.shape, b.shape, c.shape)
    _, d0, _, _ = VBMF.EVBMF(b)
    _, d1, _, _ = VBMF.EVBMF(c)
    print(d0)
    print("hello")
    ranks = [d0.shape[0], d1.shape[1]]
    print(ranks)
    exit()

    return ranks


def standard_to_tucker(standardNetwork, sess, conv_ranks, fc_ranks, num_iterations=1000, learning_rate=0.001):
    """ Accepts a standard CNN and converts it to Tucker format

        :param
            standardNetwork: Standard Network that has already been trained
            sess: Current tf.Session
            conv_ranks: Ranks for the convolutional layers
            fc_ranks: ... for the fully connected layers
            num_iterations: Number of iterations for reducing reconstruction error for each layer
            learning_rate: Learning rate for the gradient descent

        :return
            tuckerNetwork: Same architecture as standardNetwork
            sess: The updated session, containing the Tucker graph
            merged: Merged summaries """

    assert isinstance(standardNetwork, StandardNetwork)

    # All the weights across all layers
    weights = standardNetwork.get_weights()

    # Instantiate and initialise the network - using the same architecture
    tuckerNetwork = TuckerNet(standardNetwork.get_architecture())
    print("Number of parameters before = {}".format(standardNetwork.num_parameters()))

    # Now build initialise the Tucker network
    tuckerNetwork.build(conv_ranks, fc_ranks, "TuckerNet1")

    # The weights we want to update
    tucker_weights = tuckerNetwork.get_weights()

    # Go through each conv or fc layer and perform SGD to provide minimal reconstruction
    # error between the combined factors and the original weight tensor
    num_layers = standardNetwork.get_num_layers()

    # A simpler optimizer, e.g. SGD may be more appropriate
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Only need to initialize the Tucker network weights
    tucker_tfvars = [v for v in tf.trainable_variables() if "TuckerNet1" in v.name]
    init_op = tf.variables_initializer(var_list=tucker_tfvars)
    sess.run(init_op)

    for layer_idx in range(num_layers):
        cur_layer = standardNetwork.get_architecture().get_layer(layer_idx)
        if isinstance(cur_layer, ConvLayer) or isinstance(cur_layer, FullyConnectedLayer):

            # Get this current layer weights (includes bias)
            layer_weights = weights.get_layer_weights(layer_idx=layer_idx)

            # Ground truth
            y = layer_weights["kernel"]

            # Output node (reconstructed weight)
            tucker_layer_weights = tucker_weights.get_layer_weights(layer_idx=layer_idx)
            y_hat = tucker_layer_weights["graph"].combine()
            # Make sure the same shape
            y_hat = tf.reshape(y_hat, y.shape)

            # Some Tensorboard summaries
            tf.summary.histogram('original_weight_{}'.format(layer_idx), y)
            tf.summary.histogram('converted_weight_{}'.format(layer_idx), y_hat)
            tf.summary.histogram('original_bias_{}'.format(layer_idx), layer_weights["bias"])
            tf.summary.histogram('converted_bias_{}'.format(layer_idx), tucker_layer_weights["bias"])

            # Ok, long story, but here goes.
            # Exposing a set_bias function doesn't seem to make sense and exposing self.weights breaks a lot
            # of separation that is quite nice to have. Instead just add a, somewhat pointless, loss term
            # that makes the biases equal

            loss = tf.losses.mean_squared_error(y, y_hat) + tf.losses.mean_squared_error(layer_weights["bias"],
                                                                                         tucker_layer_weights["bias"])

            tf.summary.scalar('mse_loss_{}'.format(layer_idx), loss)

            # Only update the weights in the Tucker Network
            step = optimizer.minimize(loss, var_list=tucker_tfvars)

            # These will be the weights from the ADAM optimizer. They need to be initialised before sess.run
            sess.run(tf.variables_initializer(optimizer.variables()))

            # Use simple SGD to find a reasonable estimate
            # Perform N iterations to reduce reconstruction loss
            for i in range(num_iterations):
                _, mse_loss = sess.run([step, loss])
                # Loss after every iteration
                # print("mse_loss: {}".format(np.average(mse_loss)))

            # Loss after the full gradient descent on this layer
            print("mse_loss: {}".format(np.average(mse_loss)))

    # Merge all the summaries
    merged = tf.summary.merge_all()

    # Should be a reasonable approximation
    return tuckerNetwork, sess, merged


def tucker_to_standard(tuckerNetwork):
    # TODO: Simply reconstruct the core tensors and return result, don't need to minimise reconstruction error
    #       or anything
    assert isinstance(tuckerNetwork, TuckerNet)


if __name__ == "__main__":
    """ Test examples """
    from Architectures.architectures import CIFAR100Example

    architecture = CIFAR100Example()
    model = StandardNetwork(architecture=architecture)
    model.build("StandardNetwork1")

    tuckerNetwork = standard_to_tucker(model)
