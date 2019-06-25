""" Convert a network between types """

import VBMF
import numpy as np
import tensorflow as tf
from Layers.impl.core import ConvLayer, FullyConnectedLayer
from Layers.layer import ILayer
from Weights.weights import Weights

import tensorly as tl
# If use tensorflow, it gets confusing with eager execution in parts
tl.set_backend('numpy')


def estimate_ranks(weights):
    """ Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """

    # TODO: Does not quite work
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


def convert_layers(sess, input_weights, reference_weights, num_iterations=1000, learning_rate=0.001):
    """

    :param sess: Current session for the reference weights
    :param input_weights: Input weights
    :param reference_weights: Reference weights
    :param num_iterations:
    :param learning_rate:
    :return: Updated input weights
    """

    # Convert(?) to tfvar format
    y_hat = Weights.extract_tf_weights(input_weights)
    y = Weights.extract_tf_weights(reference_weights)

    # Must be of the same type
    assert(y_hat.fields == y.fields)
    fields = y._fields

    loss = 0
    # Loss is the error between all the components/weights of this layer
    for name in fields:
        loss += tf.losses.mean_squared_error(getattr(y, name), getattr(y_hat, name))
        # Some Tensorboard summaries
        tf.summary.histogram(f"inp_{name}", getattr(y_hat, name))
        tf.summary.histogram(f"ref_{name}", getattr(y, name))

    # A simpler optimizer, e.g. SGD may be more appropriate
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    step = optimizer.minimize(loss, var_list=list(y_hat))

    init_op = tf.variables_initializer(var_list=list(y_hat))
    sess.run(init_op)

    # Use simple SGD to find a reasonable estimate
    # Perform N iterations to reduce reconstruction loss
    for i in range(num_iterations):
        _, mse_loss = sess.run([step, loss])
        # Loss after every iteration
        print(f"mse_loss: {np.average(mse_loss)}")

    # Merge all the summaries
    merged = tf.summary.merge_all()

    return input_weights, merged


if __name__ == '__main__':
    tf.reset_default_graph()

