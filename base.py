import tensorflow as tf
import random
import string

print(tf.__version__)

""" Generic functions """

l2_reg = tf.contrib.layers.l2_regularizer(scale=1.0)


def variance_regularizor(scale):
    """ Variance across all dimensions/axis """
    # See: https://github.com/tensorflow/tensorflow/blob/a26d43a5f8e06f5236d4117bbda702fac26ae93a/tensorflow/contrib/layers/python/layers/regularizers.py#L37

    if scale == 0:
        return lambda _: None

    def var(weight):
        v = 0
        # Can be tidier if put all axis in tf.nn.moments
        for i, d in enumerate(weight.get_shape()):
            # Returns mean, variance
            v += tf.math.reduce_sum(tf.nn.moments(weight, axes=i)[1])

        return -1 * tf.multiply(scale, v)

    return var


def tfvar_size(tfvar):
    """ Calculate the size (number of parameters) of a tf.Variable """
    if not tfvar:
        # e.g. offset, scale is None when affine BN layers
        return 0

    # Loop through all dimensions
    i = 1
    for n in tfvar.get_shape().as_list():
        i *= n

    return i


def l2_loss_sum(list_o_tensors):
    """ Pass in an array of tensors and return the total/summed L2 loss"""
    return tf.add_n([tf.nn.l2_loss(t) for t in list_o_tensors])


def random_string(string_length=10):
    """ Generate a random string of fixed length e.g. ptmihemlzj """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(string_length))


def visualise_conv_kernel(kernel):
    """ Accepts W,H,N,C convolutional kernel and shows 3d slice plot """
    import matplotlib.pyplot as plt
    #TODO: do it
