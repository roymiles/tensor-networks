import tensorflow as tf


class Layer:
    """ Most generic abstract class """

    def __init__(self):
        pass

    def num_parameters(self):
        return 0

    def __call__(self):
        raise Exception("Must override call method")


class WeightLayer(Layer):
    """ All layer types that contain weights """

    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    def num_parameters(self):
        # Override parent function in this case
        n = 0
        for size in self._shape:
            n += size

        return n

    def get_shape(self):
        return self._shape


class ConvLayer(WeightLayer):
    def __init__(self, shape, strides=[1, 1, 1, 1]):
        super().__init__(shape)
        self._strides = strides

    def get_strides(self):
        return self._strides

    def __call__(self, input, kernel):
        return tf.nn.conv2d(input, kernel, strides=self._strides, padding="SAME")


class FullyConnectedLayer(WeightLayer):
    def __init__(self, shape):
        super().__init__(shape)

    def __call__(self, input, kernel):
        net = tf.layers.flatten(input)
        net = tf.linalg.matmul(net, kernel)

        return net


class PoolingLayer(Layer):
    def __init__(self, shape):
        """ In this case shape is the receptive field size to average over """
        super().__init__()
        self._shape = shape

    def get_pool_size(self):
        return self._shape


class AveragePoolingLayer(PoolingLayer):
    def __init__(self, shape):
        super().__init__(shape)

    def __call__(self, input):
        return tf.nn.avg_pool(input, PoolingLayer.get_pool_size(), strides=[1, 1, 1, 1], padding="SAME")


class MaxPoolingLayer(PoolingLayer):
    def __init__(self, shape):
        super().__init__(shape)

    def __call__(self, input):
        return tf.nn.max_pool(input, PoolingLayer.get_pool_size(), strides=[1, 1, 1, 1], padding="SAME")


class DropoutLayer(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self._keep_prob = keep_prob

    def __call__(self, input):
        return tf.nn.dropout(input, self._keep_prob)


class BatchNormalisationLayer(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.layers.batch_normalization(input)


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.relu(input)


class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.nn.softmax(input)


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return tf.layers.flatten(input)

