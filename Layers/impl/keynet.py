from Layers.layer import ILayer
import tensorflow as tf


# --------- Non standard layers --------- #
class GaussianSmoothLayer(ILayer):
    def __init__(self):
        from KeyNet.model.keynet_architecture import gaussian_multiple_channels
        gaussian_avg = gaussian_multiple_channels(1, 1.5)
        self._gaussian_avg = tf.constant(gaussian_avg, name='Gaussian_avg')

    def get_filter(self):
        return self._gaussian_avg

    def __call__(self, input, **kwargs):
        # Assume a single image
        return tf.nn.conv2d(input, self._gaussian_avg, strides=[1, 1, 1, 1], padding='SAME')


class PyramidScaleSplitLayer(ILayer):
    """ Let P = pyramid_levels, S = scaling_factor
        Scales images P times using S
        The result is an array of scaled images """
    def __init__(self, pyramid_levels, scaling_factor):
        super().__init__()
        self._pyramid_levels = pyramid_levels
        self._scaling_factor = scaling_factor

    def get_pyramid_levels(self):
        return self._pyramid_levels

    def get_scaling_factor(self):
        return self._scaling_factor

    def __call__(self, input, **kwargs):
        scaled_images = []
        dim_float = tf.cast(kwargs['dimension_image'], tf.float32)
        for idx_level in range(self._pyramid_levels):
            image_resized = tf.image.resize_images(input,
                                                   size=tf.cast((dim_float[1] / (self._scaling_factor ** idx_level),
                                                                 dim_float[2] / (self._scaling_factor ** idx_level)),
                                                                tf.int32), method=0)

            scaled_images.append(image_resized)

        return scaled_images


class PyramidScaleCombineLayer(ILayer):
    """ Inverse of PyramidScaleSplitLayer
        This layers resizes and then concatenates the feature maps """

    def __init__(self):
        super().__init__()

    def __call__(self, input, **kwargs):
        resized_images = []
        dim_float = tf.cast(kwargs['dimension_image'], tf.float32)
        for image in input:
            image_resized = tf.image.resize_images(image, size=tf.cast((dim_float[1], dim_float[2]), tf.int32),
                                                   method=0)

            resized_images.append(image_resized)

        return tf.concat(resized_images, axis=3)


class HandcraftedFeaturesLayer(ILayer):
    def __init__(self):
        from KeyNet.model.keynet_architecture import create_derivatives_kernel
        super().__init__()

        # Build the filters
        # Sobel derivatives
        kernel_x, kernel_y = create_derivatives_kernel()
        self._dx_filter = tf.constant(kernel_x, name='kernel_filter_dx')
        self._dy_filter = tf.constant(kernel_y, name='kernel_filter_dy')

    def __call__(self, input, **kwargs):
        """ Assume we receive a pyramid of images """

        features = []

        for image in input:
            # Sobel_conv_derivativeX
            dx = tf.nn.conv2d(image, self._dx_filter, strides=[1, 1, 1, 1], padding='SAME')
            dxx = tf.nn.conv2d(dx, self._dx_filter, strides=[1, 1, 1, 1], padding='SAME')
            dx2 = tf.multiply(dx, dx)

            # Sobel_conv_derivativeY
            dy = tf.nn.conv2d(image, self._dy_filter, strides=[1, 1, 1, 1], padding='SAME')
            dyy = tf.nn.conv2d(dy, self._dy_filter, strides=[1, 1, 1, 1], padding='SAME')
            dy2 = tf.multiply(dy, dy)

            dxy = tf.nn.conv2d(dx, self._dy_filter, strides=[1, 1, 1, 1], padding='SAME')

            dxdy = tf.multiply(dx, dy)
            dxxdyy = tf.multiply(dxx, dyy)
            dxy2 = tf.multiply(dxy, dxy)

            # Concatenate Handcrafted Features
            f = tf.concat([dx, dx2, dxx, dy, dy2, dyy, dxdy, dxxdyy, dxy], axis=3)
            features.append(f)

        return features


class ConvLayer(ILayer):
    """ Standard convolutional layer, except we assume pyramid input features """
    def __init__(self, shape, strides=[1, 1, 1, 1], use_bias=True, padding="SAME"):
        super().__init__()
        self._shape = shape
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias

    def get_shape(self):
        return self._shape

    def using_bias(self):
        return self._use_bias

    def get_strides(self):
        return self._strides

    def __call__(self, images, kernel, bias=None):

        features = []

        for image in images:
            net = tf.nn.conv2d(image, kernel, strides=self._strides, padding=self._padding)

            if bias:
                net = tf.nn.bias_add(net, bias)

            features.append(net)

        return features
