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
            AveragePoolingLayer(shape=[7, 7]),

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
    def __init__(self):
        network = [
            # NOTE: Comments are for input size
            ConvLayer(shape=[3, 3, 3, 32]),
            BatchNormalisationLayer(),
            ReLU(),
            ConvLayer(shape=[3, 3, 32, 32]),
            BatchNormalisationLayer(),
            ReLU(),
            MaxPoolingLayer(shape=[2, 2]),
            DropoutLayer(0.25),

            ConvLayer(shape=[3, 3, 32, 64]),
            BatchNormalisationLayer(),
            ReLU(),
            ConvLayer(shape=[3, 3, 64, 64]),
            BatchNormalisationLayer(),
            ReLU(),
            MaxPoolingLayer(shape=[2, 2]),
            DropoutLayer(0.25),

            # 4096 = 8 x 8 x 64
            Flatten(),
            FullyConnectedLayer(shape=[4096, 512]),
            ReLU(),
            DropoutLayer(0.5),
            FullyConnectedLayer(shape=[512, 10]),
            SoftMax()
        ]

        super().__init__(network)
