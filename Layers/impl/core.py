import tensorflow as tf
from Layers.layer import ILayer
from Networks.weights import CreateWeights

STANDARD = 1
SANDBOX = 2


class ConvLayer(ILayer):
    def __init__(self, shape, build_method=STANDARD, strides=(1, 1), use_bias=True, padding="SAME",
                 kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None):
        super().__init__()

        px = strides[0]
        py = strides[1]
        self._strides = [1, px, py, 1]
        self._build_method = build_method

        self._shape = shape
        self._padding = padding
        self._use_bias = use_bias

        # Can't be bothered to make getters for these, just make them public
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def create_weights(self):
        if self._build_method == STANDARD:
            return CreateWeights.Core.convolution
        else:
            raise Exception(f"Unknown network, Unable to create the weights for this layer: {ConvLayer}")

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def get_strides(self):
        return self._strides

    def __call__(self, input, kernel, bias=None):
        net = tf.nn.conv2d(input, kernel, strides=self._strides, padding=self._padding)

        if bias:
            net = tf.nn.bias_add(net, bias)

        return net


class ConvLayerConstant(ILayer):
    """ Same as ConvLayer, but fixed, constant filter """
    def __init__(self, kernel, build_method=STANDARD, strides=[1, 1, 1, 1], padding="SAME"):
        super().__init__()
        self._kernel = kernel
        self._build_method = build_method
        self._strides = strides
        self._padding = padding

    def create_weights(self):
        if self._build_method == STANDARD:
            return CreateWeights.Core.convolution
        else:
            raise Exception(f"Unknown network, Unable to create the weights for this layer: {ConvLayer}")

    def get_kernel(self):
        return self._kernel

    def get_strides(self):
        return self._strides

    def __call__(self, input):
        return tf.nn.conv2d(input, self._kernel, strides=self._strides, padding=self._padding)


class DepthwiseConvLayer(ILayer):
    """ Depthwise convolution
        NOTE: Pointwise convolution uses standard conv layer """
    def __init__(self, shape, build_method=STANDARD, strides=(1, 1), use_bias=True, padding="SAME",
                 kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None):
        super().__init__()

        px = strides[0]
        py = strides[1]
        self._strides = [1, px, py, 1]
        self._build_method = build_method

        self._shape = shape
        self._padding = padding
        self._use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def create_weights(self):
        if self._build_method == STANDARD:
            return CreateWeights.Core.depthwise_convolution
        else:
            raise Exception(f"Unknown network, Unable to create the weights for this layer: {DepthwiseConvLayer}")

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def get_strides(self):
        return self._strides

    def __call__(self, input, kernel, bias=None):
        # Depthwise convolution
        net = tf.nn.depthwise_conv2d(input, kernel, strides=self._strides, padding=self._padding)

        if bias:
            net = tf.nn.bias_add(net, bias)

        return net


class BiasLayerConstant(ILayer):
    """ Fixed, constant bias added to a layer """
    def __init__(self, bias):
        super().__init__()
        self._bias = bias

    def get_bias(self):
        return self._bias

    def __call__(self, input):
        return tf.nn.bias_add(input, self._bias)


class FullyConnectedLayer(ILayer):
    def __init__(self, shape, build_method=STANDARD, use_bias=True, kernel_initializer=tf.glorot_normal_initializer(),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None):

        super().__init__()
        self._shape = shape
        self._build_method = build_method
        self._use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def create_weights(self):
        if self._build_method == STANDARD:
            return CreateWeights.Core.fully_connected
        else:
            raise Exception(f"Unknown network, Unable to create the weights for this layer: {FullyConnectedLayer}")

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def __call__(self, input, kernel, bias=None):
        # net = tf.linalg.matmul(input, kernel)
        net = tf.tensordot(input, kernel, axes=[1, 0])

        if bias:
            net = tf.nn.bias_add(net, bias)

        return net


# Alias name
Dense = FullyConnectedLayer


class PoolingLayer(ILayer):
    def __init__(self, pool_size=(2, 2)):
        """ In this case shape is the receptive field size to average over """
        px = pool_size[0]
        py = pool_size[1]
        self._ksize = [1, px, py, 1]
        self._strides = [1, px, py, 1]

    def get_ksize(self):
        return self._ksize

    def get_strides(self):
        return self._strides


class AveragePoolingLayer(PoolingLayer):
    def __init__(self, pool_size):
        super().__init__(pool_size)

    def __call__(self, input):
        return tf.nn.avg_pool(input, ksize=super(AveragePoolingLayer, self).get_ksize(),
                              strides=super(AveragePoolingLayer, self).get_strides(), padding="SAME")


class MaxPoolingLayer(PoolingLayer):
    def __init__(self, pool_size):
        super().__init__(pool_size)

    def __call__(self, input):
        return tf.nn.max_pool(input, ksize=super(MaxPoolingLayer, self).get_ksize(),
                              strides=super(MaxPoolingLayer, self).get_strides(), padding="SAME")


class GlobalAveragePooling:
    """ Pool over entire spatial dimensions"""
    def __init__(self, keep_dims=True):
        self._keep_dims = keep_dims

        super().__init__()

    def __call__(self, input):
        return tf.reduce_mean(input, [1, 2], keep_dims=self._keep_dims, name='global_pool')


class DropoutLayer(ILayer):
    def __init__(self, rate):
        # rate defines the fraction of input units to drop
        super().__init__()
        self._rate = rate

    def __call__(self, input):
        return tf.nn.dropout(input, rate=self._rate)


class BatchNormalisationLayer(ILayer):
    def __init__(self, affine=True):

        """ If affine is False, the scale and offset parameters won't be used
            When affine=False the output of BatchNorm is equivalent to considering gamma=1 and beta=0 as constants. """
        super().__init__()
        self._affine = affine

    def __call__(self, input, is_training):
        return tf.layers.batch_normalization(input, training=is_training)


class ReLU(ILayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.relu(input)


class ReLU6(ILayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.relu6(input)


class hswish(ILayer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x * tf.nn.relu6(x + 3) / 6


class SoftMax(ILayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.softmax(input)


class Flatten(ILayer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.layers.flatten(input)


""" Multilayer classes e.g. residual layers etc """


class MobileNetV2BottleNeck(ILayer):
    def __init__(self, k, t, c, build_method=STANDARD, strides=(1, 1)):
        """
        See: https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c

        :param k: Number of input channels
        :param t: Expansion factor
        :param c: Number of output channels
        :param strides: Depthwise stride
        """
        super().__init__()

        px = strides[0]
        py = strides[1]
        self._strides = [1, px, py, 1]
        self._build_method = build_method
        self._k = k
        self._t = t
        self._c = c

        super().__init__()

    def create_weights(self):
        if self._build_method == STANDARD:
            return CreateWeights.Core.mobilenetv2_bottleneck
        else:
            raise Exception(f"Unknown network, Unable to create the weights for this layer: {MobileNetV2BottleNeck}")

    def __call__(self, input, expansion_kernel, expansion_bias, depthwise_kernel, depthwise_bias,
                 projection_kernel, projection_bias):

        # Just use tf.layers ...
        # expansion_mean, expansion_variance, expansion_scale, expansion_offset,
        # depthwise_mean, depthwise_variance, depthwise_scale, depthwise_offset,
        # projection_mean, projection_variance, projection_scale, projection_offset):

        # Expansion layer
        net = tf.nn.conv2d(input, expansion_kernel, strides=[1, 1, 1, 1], padding="SAME")
        net = tf.nn.bias_add(net, expansion_bias)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu6(net)

        # Depthwise layer
        net = tf.nn.depthwise_conv2d(net, depthwise_kernel, strides=self._strides, padding="SAME")
        net = tf.nn.bias_add(net, depthwise_bias)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu6(net)

        # Projection layer (linear)
        net = tf.nn.conv2d(net, projection_kernel, strides=[1, 1, 1, 1], padding="SAME")
        net = tf.nn.bias_add(net, projection_bias)
        net = tf.layers.batch_normalization(net)

        # Only residual add when strides is 1
        is_residual = True if self._strides == [1, 1, 1, 1] else False
        is_conv_res = False if net.get_shape().as_list()[3] == input.get_shape().as_list()[3] else True

        if is_residual:

            if is_conv_res:
                # See: https://github.com/MG2033/MobileNet-V2/blob/master/layers.py
                # If not matching channels, place a 1x1 convolution to ensure match
                x = tf.layers.conv2d(input, net.get_shape().as_list()[3], (1, 1), use_bias=False,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(),
                                     name="fix-channel-mismatch")
                return net + x
            else:
                return net + input
        else:
            return net

    def get_k(self):
        return self._k

    def get_t(self):
        return self._t

    def get_c(self):
        return self._c

    def get_strides(self):
        return self._strides
