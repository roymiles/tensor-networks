from Architectures.architectures import IArchitecture
import Layers.impl.core as core_layers
import Layers.impl.keynet as keynet_layers
import tensorflow as tf


class KeyNet(IArchitecture):
    # Axel's Key.Net
    def __init__(self):
        network = [
            keynet_layers.GaussianSmoothLayer(),

            # Makes a parallel split
            keynet_layers.PyramidScaleSplitLayer(pyramid_levels=3, scaling_factor=1.2),
            keynet_layers.HandcraftedFeaturesLayer(),
            keynet_layers.ConvLayer(shape=(5, 5, 9, 8), strides=[1, 1, 1, 1], use_bias=False),
            keynet_layers.ConvLayer(shape=(5, 5, 8, 8), strides=[1, 1, 1, 1], use_bias=False),
            keynet_layers.ConvLayer(shape=(5, 5, 8, 8), strides=[1, 1, 1, 1], use_bias=False),
            # Combines the parallel split
            keynet_layers.PyramidScaleCombineLayer(),

            # Back to standard layers
            core_layers.BatchNormalisationLayer(24, affine=False),
            core_layers.ConvLayer(shape=[5, 5, 24, 1])
        ]

        super().__init__(network)

        MSIP_sizes = [8, 16, 24, 32, 40]
        self.create_kernels(MSIP_sizes, "KeyNet")

    def create_kernels(self, MSIP_sizes, name_scope):
        # create_kernels
        self.kernels = {}

        # Grid Indexes for MSIP
        for ksize in MSIP_sizes:
            from KeyNet.model.keynet_architecture import ones_multiple_channels, grid_indexes, linear_upsample_weights
            ones_kernel = ones_multiple_channels(ksize, 1)
            indexes_kernel = grid_indexes(ksize)
            upsample_filter_np = linear_upsample_weights(int(ksize / 2), 1)

            self.ones_kernel = tf.constant(ones_kernel, name=name_scope +'_Ones_kernel_'+str(ksize))
            self.kernels['ones_kernel_'+str(ksize)] = self.ones_kernel

            self.upsample_filter_np = tf.constant(upsample_filter_np, name=name_scope+'_upsample_filter_np_'+str(ksize))
            self.kernels['upsample_filter_np_'+str(ksize)] = self.upsample_filter_np

            self.indexes_kernel = tf.constant(indexes_kernel, name=name_scope +'_indexes_kernel_'+str(ksize))
            self.kernels['indexes_kernel_'+str(ksize)] = self.indexes_kernel

    def get_kernels(self):
        return self.kernels
