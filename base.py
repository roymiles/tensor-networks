import tensorflow as tf

""" Generic functions """

# NOTE: Wide compression suggests to use Gaussian initialization
initializer = tf.contrib.layers.variance_scaling_initializer()
