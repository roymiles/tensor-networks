import tensorflow as tf

""" Generic functions """

# NOTE: Wide compression suggests to use Gaussian initialization
initializer = tf.contrib.layers.variance_scaling_initializer()


def tfvar_size(tfvar):
    # Loop through all dimensions
    i = 1
    for n in tfvar.get_shape().as_list():
        i *= n

    return i
