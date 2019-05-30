from Architectures.architectures import IArchitecture
from Layers.impl.core import *
import tensorflow as tf

# Configure the learning rate using an exponential decay.
num_epochs_per_decay = 2.5
dataset_size = 1271167
batch_size = 64
decay_steps = int(dataset_size / batch_size * num_epochs_per_decay)
initial_learning_rate = 1e-4
_LEARNING_RATE_DECAY_FACTOR = 0.94

learning_rate = tf.train.exponential_decay(
    initial_learning_rate,
    tf.train.get_or_create_global_step(),
    decay_steps,
    _LEARNING_RATE_DECAY_FACTOR,
    staircase=True)

training_params = {
    "batch_size": 64,
    "optimizer": tf.train.GradientDescentOptimizer(learning_rate),

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

    weight_decay = 0.00004
    stddev = 0.09
    batch_norm_decay = 0.9997
    batch_norm_epsilon = 0.001

    def DepthSepConv(self, shape, stride, depth):
        # Depth is pretty much shape[3] (if included)
        w = shape[0]
        h = shape[1]
        c = shape[2]
        depth_multiplier = 1

        # By default, don't regularise depthwise filters
        sequential = [
            DepthwiseConvLayer(shape=[w, h, c, depth_multiplier], strides=(stride, stride),
                               kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev)),
            ConvLayer(shape=[1, 1, c, depth], kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay)),
            # Not managed to integrate moving average decay
            BatchNormalisationLayer(num_features=depth, variance_epsilon=self.batch_norm_epsilon),
            ReLU()
        ]

        return sequential

    def __init__(self, num_classes):
        network = [
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 32], strides=(2, 2)),
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

            GlobalAveragePooling(),
            DropoutLayer(0.999),
            # Logits
            ConvLayer(shape=[3, 3, 1024, num_classes]),
        ]

        super().__init__(network)
