import numpy as np
import random


def random_horizontal_flip(images):
    # Either the same, or a horizontal flip (aka image[::-1])
    # n, w, h, c
    # slice(start, stop, end)
    flips = [(slice(None), slice(None), slice(None)), (slice(None), slice(None, None, -1), slice(None))]

    return np.array([
        # Loop over each image and apply random flip
        img[random.choice(flips)] for img in images
    ])


def normalize(images, mean, std):
    """ Mean and std are for each channel RGB
        NOTE USED """

    imgs = np.array(images)
    for i in range(3):
        images[:, :, :, i] = (imgs[:, :, :, i] - mean[i]) / std[i]

    return images
