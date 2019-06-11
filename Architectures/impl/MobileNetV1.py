from Architectures.architectures import IArchitecture
from Layers.impl.core import *
import tensorflow as tf

learning_rate = tf.placeholder(tf.float64, shape=[])
training_params = {
    "initial_learning_rate": 1e-4,
    "learning_rate": learning_rate,

    "num_epochs_per_decay": 3,
    "lr_decay": 0.94,

    "batch_size": 64,
    "optimizer": tf.train.GradientDescentOptimizer(learning_rate=learning_rate),

    # These hyperparameters control the compression
    # of the convolutional and fully connected weights
    "conv_ranks": {
        0: [6, 8, 16],
        1: [6, 16, 32],
        2: [6, 32, 64],
        3: [6, 64, 64],
        4: [6, 64, 128],
        5: [6, 128, 128],
        6: [6, 128, 256],

        7: [6, 256, 256],
        8: [6, 256, 256],
        9: [6, 256, 256],
        10: [6, 256, 256],
        11: [6, 256, 256],

        12: [6, 256, 512],
        13: [6, 512, 512]
    },
    "fc_ranks": {
        16: [512, 256]
    }

}


class MobileNetV1(IArchitecture):
    """ This is the original MobileNet architecture for ImageNet
        Of course, the convolutional layers are replaced with depthwise separable layers """

    def DepthSepConv(self, shape, stride, depth):
        # Depth is pretty much shape[3] (if included)
        w = shape[0]
        h = shape[1]
        c = shape[2]
        depth_multiplier = 1

        # By default, don't regularise depthwise filters
        sequential = [
            DepthwiseConvLayer(shape=[w, h, c, depth_multiplier], strides=(stride, stride), use_bias=False),
            BatchNormalisationLayer(num_features=c * depth_multiplier),
            ReLU(),
            # Pointwise
            ConvLayer(shape=[1, 1, c * depth_multiplier, depth], use_bias=False),
            # Not managed to integrate moving average decay
            BatchNormalisationLayer(num_features=depth),
            ReLU()
        ]

        return sequential

    def __init__(self, num_classes):
        network = [
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 32], strides=(2, 2)),
            ReLU(),

            *self.DepthSepConv(shape=[3, 3, 32], stride=1, depth=64),
            *self.DepthSepConv(shape=[3, 3, 64], stride=2, depth=128),
            *self.DepthSepConv(shape=[3, 3, 128], stride=1, depth=128),
            *self.DepthSepConv(shape=[3, 3, 128], stride=2, depth=256),
            *self.DepthSepConv(shape=[3, 3, 256], stride=1, depth=256),
            *self.DepthSepConv(shape=[3, 3, 256], stride=2, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=1, depth=512),
            *self.DepthSepConv(shape=[3, 3, 512], stride=2, depth=1024),
            *self.DepthSepConv(shape=[3, 3, 1024], stride=1, depth=1024),

            ConvLayer(shape=[1, 1, 1024, 1024], use_bias=False),
            GlobalAveragePooling(keep_dims=False),
            FullyConnectedLayer(shape=[1024, num_classes])
        ]

        super().__init__(network)
