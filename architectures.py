from layers import *


class Architecture:
    def __init__(self, network):
        # Array of sequential layers
        self._network = network

    def num_parameters(self):
        """ Get the number of parameters for an entire architecture """
        n = 0
        for layer in self._network:
            n += layer.num_parameters()

        return n

    def num_layers(self):
        return len(self._network)

    def get_layer(self, layer_idx):
        return self._network[layer_idx]


class MobileNetV1(Architecture):
    """ This is the original MobileNet architecture for ImageNet """
    def __init__(self):
        network = [
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 32], strides=[1, 2, 2, 1]),  # 224 x 224 x 3
            ConvLayer(shape=[3, 3, 32, 64]),  # 112 x 112 x 32
            ConvLayer(shape=[3, 3, 64, 128], strides=[1, 2, 2, 1]),  # 112 x 112 x 64
            ConvLayer(shape=[3, 3, 128, 128]),  # 56 x 56 x 128
            ConvLayer(shape=[3, 3, 128, 256], strides=[1, 2, 2, 1]),  # 56 x 56 x 128
            ConvLayer(shape=[3, 3, 256, 256]),  # 28 x 28 x 256
            ConvLayer(shape=[3, 3, 256, 512], strides=[1, 2, 2, 1]),  # 28 x 28 x 256

            # Repeated 5x
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 512, 512]),  # 14 x 14 x 512

            # Confused about stride here (from paper they are both = 2)
            ConvLayer(shape=[3, 3, 512, 1024], strides=[1, 2, 2, 1]),  # 14 x 14 x 512
            ConvLayer(shape=[3, 3, 1024, 1024]),  # 7 x 7 x 512

            # Global average pooling
            AveragePoolingLayer(shape=[1, 7, 7, 1]),

            # Finally a fully connected layer (1000 classes)
            FullyConnectedLayer(shape=[1024, 1000])
        ]

        super().__init__(network)


class AlexNet(Architecture):
    def __init__(self):
        network = [
            # TODO: Finish
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 96]),  # 227x227x3
            ConvLayer(shape=[3, 3, 96, 256]),
            ConvLayer(shape=[3, 3, 256, 384], strides=[1, 2, 2, 1]),
            ConvLayer(shape=[3, 3, 384, 384]),
            ConvLayer(shape=[3, 3, 384, 256], strides=[1, 2, 2, 1]),
            ConvLayer(shape=[3, 3, 256, 256]),
            ConvLayer(shape=[3, 3, 256, 512], strides=[1, 2, 2, 1]),

        ]

        super().__init__(network)


class CIFAR100Example(Architecture):
    # See: https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py
    # ... CIFAR100, but 10 logits dimension?
    def __init__(self):
        network = [
            ConvLayer(shape=[3, 3, 1, 32]),
            BatchNormalisationLayer(32),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32]),
            BatchNormalisationLayer(32),
            ReLU(),
            MaxPoolingLayer(shape=[1, 2, 2, 1]),
            DropoutLayer(0.25),

            ConvLayer(shape=[3, 3, 32, 64]),
            BatchNormalisationLayer(64),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64]),
            BatchNormalisationLayer(64),
            ReLU(),
            MaxPoolingLayer(shape=[1, 2, 2, 1]),
            DropoutLayer(0.25),

            Flatten(),
            FullyConnectedLayer(shape=[3136, 512]),
            ReLU(),
            DropoutLayer(0.5),
            FullyConnectedLayer(shape=[512, 10])
        ]

        super().__init__(network)


class HardNet(Architecture):
    # See: https://github.com/DagnyT/hardnet
    def __init__(self):
        network = [
            ConvLayer(shape=[3, 3, 1, 32], name="conv1", use_bias=False),
            BatchNormalisationLayer(32, name="bn1", affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32], name="conv2", use_bias=False),
            BatchNormalisationLayer(32, name="bn2", affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 64], name="conv3", use_bias=False),
            BatchNormalisationLayer(64, name="bn3", affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64], name="conv4", use_bias=False),
            BatchNormalisationLayer(64, name="bn4", affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 128], name="conv5", use_bias=False),
            BatchNormalisationLayer(128, name="bn5", affine=False),
            ReLU(),
            ConvLayer(shape=[3, 3, 128, 128], name="conv6", use_bias=False),
            BatchNormalisationLayer(128, name="bn6", affine=False),
            ReLU(),
            DropoutLayer(0.3),
            ConvLayer(shape=[8, 8, 128, 128], name="conv7", use_bias=False, padding="VALID"),
            BatchNormalisationLayer(128, name="bn7", affine=False)
        ]

        super().__init__(network)

    @staticmethod
    def input_norm(x, eps=1e-6):
        x_flatten = tf.layers.flatten(x)
        x_mu, x_std = tf.nn.moments(x_flatten, axes=[1])
        # Add extra dimension
        x_mu = tf.expand_dims(x_mu, axis=1)
        x_std = tf.expand_dims(x_std, axis=1)
        x_norm = (x_flatten - x_mu) / (x_std + eps)
        return tf.reshape(x_norm, shape=x.shape)

    def forward(self, input):
        # TODO: Not using Keras anymore mate
        input_norm = self.input_norm(input)
        # Need to reshape input to: [batch, in_height, in_width, in_channels]
        # Before it was: [batch, in_channels, in_height, in_width]
        new_shape = (input_norm.shape[0], input_norm.shape[2], input_norm.shape[3], input_norm.shape[1])
        input_norm = tf.reshape(input_norm, shape=new_shape)

        x_features = self.features(input_norm)
        x = tf.reshape(x_features, shape=(x_features.shape[0], -1))
        x_norm = tf.math.l2_normalize(x, axis=1)
        return x_norm
