import tensorflow as tf
from Networks.graph import Graph


def mobilenetv2_bottleneck(cur_layer, layer_idx):
    """
        Standard MobileNetV2 bottleneck layer (expansion, depthwise, linear projection and residual add)
    """
    weight_decay = 0.00004
    t = cur_layer.get_t()  # Expansion

    # Yes, letter choice is contradictory with convention here, where C is commonly input channels
    c = cur_layer.get_c()  # Number of output channels
    k = cur_layer.get_k()  # Number of input channels

    # Standard MobileNet
    expansion_kernel = tf.get_variable(f"expansion_{layer_idx}", shape=[1, 1, k, t*k],
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                       initializer=tf.keras.initializers.glorot_normal(),
                                       regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    projection_kernel = tf.get_variable(f"projection_{layer_idx}", shape=[1, 1, t*k, c],
                                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                        initializer=tf.keras.initializers.glorot_normal(),
                                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    depthwise_kernel = tf.get_variable(f"depthwise_{layer_idx}", shape=[3, 3, t*k, 1],
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS],
                                       initializer=tf.keras.initializers.glorot_normal(),
                                       regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    tf.summary.histogram("expansion_kernel", expansion_kernel)
    tf.summary.histogram("projection_kernel", projection_kernel)

    # Use biases for all the convolutions
    expansion_bias = tf.get_variable(f"expansion_bias_{layer_idx}", shape=[t * k],
                                     initializer=tf.keras.initializers.constant(0),
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

    depthwise_bias = tf.get_variable(f"depthwise_bias_{layer_idx}", shape=[t * k],
                                     initializer=tf.keras.initializers.constant(0),
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

    projection_bias = tf.get_variable(f"projection_bias_{layer_idx}", shape=[c],
                                      initializer=tf.keras.initializers.constant(0),
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

    tf.summary.histogram("expansion_bias", expansion_bias)
    tf.summary.histogram("depthwise_bias", depthwise_bias)
    tf.summary.histogram("projection_bias", projection_bias)

    return {"expansion_kernel": expansion_kernel, "expansion_bias": expansion_bias,
            "depthwise_kernel": depthwise_kernel, "depthwise_bias": depthwise_bias,
            "projection_kernel": projection_kernel, "projection_bias": projection_bias}