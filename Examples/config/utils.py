import tensorflow as tf
import json
import os
import yaml
import io
from collections import namedtuple
from Architectures.impl.MobileNetV1 import MobileNetV1
from Architectures.impl.MobileNetV2 import MobileNetV2
from Architectures.impl.CIFARExample import CIFARExample
from Architectures.impl.MNISTExample import MNISTExample
from Architectures.impl.AlexNet import AlexNet
from Architectures.impl.DenseNet import DenseNet
from transforms import aspect_preserving_resize

"""
    This file provides utility functions for using the .json/.yaml configuration files
"""


def _dict_object_hook(d): return namedtuple('X', d.keys())(*d.values())


# Convert json file to object
def json2obj(data): return json.loads(data, object_hook=_dict_object_hook)


def load_config(filename):
    """ Load training configuration """
    _, config_ext = os.path.splitext(filename)

    if config_ext == ".json":

        with open(f"{os.getcwd()}/config/{filename}") as json_file:
            d = json_file.read()
            x = json2obj(d)
            return x

    elif config_ext == ".yaml":

        with open(f"{os.getcwd()}/config/{filename}", 'r') as yaml_file:
            d = yaml_file.safe_load()
            x = _dict_object_hook(d)
            return x

    else:
        raise Exception("Unknown file type")


def get_architecture(args, ds_args):
    """
        Get the architecture based on the arguments

        :param args: Model/Training arguments
        :param ds_args: Dataset arguments
        :return: The architecture instance
    """

    name = args.architecture
    if name == "MobileNetV1":
        return MobileNetV1(args, ds_args)
    elif name == "MobileNetV2":
        return MobileNetV2(args, ds_args)
    elif name == "CIFARExample":
        return CIFARExample(args, ds_args)
    elif name == "MNISTExample":
        return MNISTExample(args, ds_args)
    elif name == "AlexNet":
        return AlexNet(args, ds_args)
    elif name == "DenseNet":
        return DenseNet(args, ds_args)
    else:
        raise Exception("Unknown architecture")


def get_optimizer(args):
    """
        Get the optimizer based on the arguments

        :param args: Object containing arguments as members, including name of optimizer, learning rate etc
        :return: Optimizer, just call .minimize on this object AND the learning rate placeholder
    """

    with tf.variable_scope("input"):
        learning_rate = tf.placeholder(tf.float64, shape=[])

    tf.summary.scalar("learning_rate", learning_rate, collections=['train'])

    opt = args.optimizer
    if opt.name == "Adam":
        return tf.train.AdamOptimizer(learning_rate=learning_rate), learning_rate
    elif opt.name == "RMSProp":
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate), learning_rate
    elif opt.name == "Momentum":
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=opt.momentum,
                                          use_nesterov=opt.use_nesterov), learning_rate
    else:
        raise Exception("Unknown optimizer")


def preprocess_images_fn(args, ds_args, is_training=True):
    """ Apply the pre-processing transforms as defined by the config """

    # 40 -> 32 for Cifar10/100
    # 256 -> 224 for ImageNet

    if is_training:
        transforms = args.pre_processing.train
    else:
        transforms = args.pre_processing.test

    # List of all the transform functions
    funcs = []

    for name, properties in transforms._asdict().items():
        # if name == "scale":
        #    # range [0, 1] scale based on maximum value
        #    funcs.append(lambda x, scale=properties.val: x / scale)
        # elif name == "normalize":
        #    # range [-1, 1] normalise with mean and std
        #    funcs.append(lambda x, mean=ds_args.mean, std=ds_args.std: normalize_images(x, mean, std))
        if name == "random_crop":
            funcs.append(lambda x, width=properties.width, height=properties.height:
                         tf.image.random_crop(x, size=[width, height, 3]))
        elif name == "random_flip_left_right":
            funcs.append(lambda x: tf.image.random_flip_left_right(x))
        elif name == "resize":
            funcs.append(lambda x, width=properties.width, height=properties.height:
                         tf.compat.v1.image.resize(x, [width, height],
                                                   method=tf.image.ResizeMethod.BILINEAR,
                                                   align_corners=False))

    # Combine the list of transforms into a single lambda expression
    def f(x):
        # reversed(funcs) if applying transforms bottom up (in config)
        for func in funcs:
            x = func(x)

        return x

    return f


def generate_unique_name(args, ds_args):
    """ Using the configuration, generate a semi unique string name for saving logs, checkpoints etc """
    unique_name = f"arch_{args.architecture}_ds_{args.dataset_name}_opt_{args.optimizer.name}"
    if hasattr(args, 'build_method'):
        unique_name += f"_build_method_{args.build_method}"

    """
    if args.method == "custom-bottleneck":
        if hasattr(args, 'ranks'):
            unique_name += "_ranks_"
            unique_name += '_'.join(str(x) for x in args.ranks)
    
        if hasattr(args, 'partitions'):
            unique_name += "_partitions_"
            unique_name += '_'.join(str(x) for x in args.partitions)
    """

    return unique_name


def anneal_learning_rate(lr, epoch, step, args, sess=None):
    """ Perform learning rate annealing, as defined by the training .json/.yaml config """
    try:
        if hasattr(args, "lr_annealing"):

            if args.lr_annealing.name == "num_epochs_decay":
                # Decay every n epochs
                if epoch % args.lr_annealing.num_epochs_decay == 0:
                    return lr * args.lr_annealing.lr_decay
            elif args.lr_annealing.name == "epoch_decay_boundaries":
                # Decay at predefined epoch boundaries
                if epoch in args.lr_annealing.epoch_decay_boundaries:
                    return lr * args.lr_annealing.lr_decay
            elif args.lr_annealing.name == "noisy_linear_cosine_decay":
                decayed_lr = tf.train.noisy_linear_cosine_decay(lr, step, args.noisy_linear_cosine_decay.decay_steps)
                return sess.run(decayed_lr)
            else:
                raise Exception("Unspecified learning rate annealing strategy")

            # Not decaying for this epoch
            return lr

    except AttributeError:
        # Not performing any lr annealing
        return lr


