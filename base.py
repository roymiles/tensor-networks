import tensorflow as tf
import random
import string

print(tf.__version__)

""" Generic functions """

l2_reg = tf.contrib.layers.l2_regularizer(scale=1.0)


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
    return ''.join(random.choice(letters) for i in range(string_length))
