from Architectures.architectures import IArchitecture
from Layers.impl.core import *


class TransitionLayer(IArchitecture):
    def __init__(self, k0):
        """
        Change feature-map sizes via convolution and pooling

        :param k0: Number of input channels
        """
        print("k0 = {}".format(k0))
        network = [
            BatchNormalisationLayer(k0),
            ReLU(),
            ConvLayer(shape=[1, 1, k0, 1]),
            AveragePoolingLayer(pool_size=(2, 2))
        ]

        super().__init__(network)


class DenseBlock(IArchitecture):
    def __init__(self, n, k0, k):
        """

        :param n: Number of 1x1 then 3x3 layers
        :param k0: Input channels
        :param k: Growth rate (is also the input channel dim k0)
        """

        c = k0
        network = []
        for _ in range(n):
            print("c = {}".format(c))
            _net = [
                BatchNormalisationLayer(c),
                ReLU(),
                ConvLayer(shape=[1, 1, c, k])
            ]

            network = network + _net

            # Growth rate
            c += k

        # Output channels
        self._c = c

        super().__init__(network)

    def get_output_dim(self):
        return self._c


class DenseNet(IArchitecture):
    def __init__(self, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0,
                 weight_decay=1e-4, classes=1000, weights_path=None):
        """ Instantiate the DenseNet architecture,
            # Arguments
                nb_dense_block: number of dense blocks to add to end
                growth_rate: number of filters to add per dense block
                nb_filter: initial number of filters
                reduction: reduction factor of transition blocks.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                classes: optional number of classes to classify images
                weights_path: path to pre-trained weights
            # Returns
                A Keras model instance.
        """
        eps = 1.1e-5

        # compute compression factor
        compression = 1.0 - reduction

        # From architecture for ImageNet (Table 1 in the paper)
        nb_filter = 64
        nb_layers = [6, 12, 32, 32]  # For DenseNet-169

