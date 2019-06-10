from Architectures.architectures import IArchitecture
from Layers.impl.core import *


# These hyperparameters control the compression
# of the convolutional and fully connected weights
conv_ranks = {
    0: [6, 32, 32],
    2: [6, 32, 32],
    6: [6, 32, 32],
    8: [6, 32, 32]
}
fc_ranks = {
    13: [52, 52],
    16: [52, 52]
}

learning_rate = tf.placeholder(tf.float64, shape=[])
training_params = {
    "initial_learning_rate": 0.01,
    "learning_rate": learning_rate,
    "batch_size": 128,

    "nb_epoch": 12,
    "lr_decay": 1e-6,

    "optimizer": tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True),
}


class CIFARExample(IArchitecture):
    # See: https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py
    def __init__(self, num_classes=100):
        # Change num_classes 10/100 depending on cifar10 or cifar100
        network = [
            # NOTE: Don't add batch normalisation layers, this messes up on 12 epochs for cifar10

            # 0:5
            ConvLayer(shape=[3, 3, 3, 32]),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32]),
            ReLU(),
            MaxPoolingLayer(pool_size=(2, 2)),
            DropoutLayer(0.25),

            # 6:11
            ConvLayer(shape=[3, 3, 32, 64]),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64]),
            ReLU(),
            MaxPoolingLayer(pool_size=(2, 2)),
            DropoutLayer(0.25),

            # 12:16
            Flatten(),
            FullyConnectedLayer(shape=[4096, 512]),
            ReLU(),
            DropoutLayer(0.5),
            FullyConnectedLayer(shape=[512, num_classes])
        ]

        super().__init__(network)
