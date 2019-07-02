import numpy as np
import random
import tensorflow as tf


def random_horizontal_flip(images):
    # Either the same, or a horizontal flip (aka image[::-1])
    # n, w, h, c
    # slice(start, stop, end)
    flips = [(slice(None), slice(None), slice(None)), (slice(None), slice(None, None, -1), slice(None))]

    return np.array([
        # Loop over each image and apply random flip
        img[random.choice(flips)] for img in images
    ])


def normalize_images(images, mean, std):
    """ Mean and std are for each channel RGB
        Accepts either a batch of images or a single image """

    imgs = np.array(images)
    for i in range(3):
        imgs[..., i] = (imgs[..., i] - mean[i]) / std[i]

    return imgs


def aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.
    Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
    Returns:
    resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    new_height, new_width = smallest_size_at_least(height, width, resize_min)

    ret = resize_image(image, new_height, new_width)
    return ret


def resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.
    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.
    Args:
        image: A 3-D image `Tensor`.
        height: The target height for the resized image.
        width: The target width for the resized image.
    Returns:
        resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    return tf.compat.v1.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)


def smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
        height: an int32 scalar tensor indicating the current height.
        width: an int32 scalar tensor indicating the current width.
        resize_min: A python integer or scalar `Tensor` indicating the size of
          the smallest side after resize.
    Returns:
        new_height: an int32 scalar tensor indicating the new height.
        new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width
