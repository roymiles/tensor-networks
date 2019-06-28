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


def _dict_object_hook(d): return namedtuple('X', d.keys())(*d.values())


# Convert json file to object
def json2obj(data): return json.loads(data, object_hook=_dict_object_hook)


_JSON = 0
_YAML = 1


def load_config(name, mode=_JSON, as_obj=True):
    """ Load training configuration """
    if mode == _JSON:
        """ Load .json format """
        with open(f"{os.getcwd()}/config/{name}") as json_file:
            if as_obj:
                # As object
                d = json_file.read()
                x = json2obj(d)
                return x
            else:
                # As dictionary
                return json.load(json_file)
    elif mode == _YAML:
        """ Load .yaml format """
        with open(f"{os.getcwd()}/config/{name}", 'r') as yaml_file:
            d = yaml_file.safe_load()
            if as_obj:
                # As object
                x = _dict_object_hook(d)
                return x
            else:
                # As dictionary
                return d
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
        :return: Optimizer, just call .minimize on this object
    """

    with tf.variable_scope("input"):
        learning_rate = tf.placeholder(tf.float64, shape=[])

    name = args.optimizer
    if name == "Adam":
        return tf.train.AdamOptimizer(learning_rate=learning_rate), learning_rate
    elif name == "RMSProp":
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate), learning_rate
    elif name == "Momentum":
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=args.momentum,
                                          use_nesterov=args.use_nesterov), learning_rate
    else:
        raise Exception("Unknown optimizer")


def preprocess_images_fn(ds_args):
    """ Data augmentation """
    # TODO: Hard coded sizes
    # 40 -> 32 for Cifar10/100
    # 256 -> 224 for ImageNet
    return lambda x: tf.image.random_crop(tf.compat.v1.image.resize(tf.image.random_flip_left_right(x),
                                                                    # [ds_args.img_height, ds_args.img_width],
                                                                    [40, 40],
                                                                    method=tf.image.ResizeMethod.BILINEAR,
                                                                    align_corners=False),
                                          size=[32, 32, 3])


def is_epoch_decay(epoch, args):
    """ Can decay depending on list or every n epochs """
    if epoch == 0:
        return False

    if hasattr(args, 'num_epochs_decay'):
        if epoch % args.num_epochs_decay == 0:
            return True
    elif hasattr(args, 'epoch_decay_boundaries'):
        if epoch in args.epoch_decay_boundaries:
            return True

    return False


def anneal_learning_rate(lr, epoch, step, args):
    """ Perform learning rate annealing, as defined by the training .json/.yaml config """
    try:
        if hasattr(args, "basic_lr_annealing"):

            # These are basic learning rate annealing strategies
            if hasattr(args.basic_lr_annealing, "num_epochs_decay"):
                # Decay every n epochs
                if epoch % args.num_epochs_decay == 0:
                    return lr * args.basic_lr_annealing.lr_decay
            elif hasattr(args, 'epoch_decay_boundaries'):
                # Decay at predefined epoch boundaries
                if epoch in args.epoch_decay_boundaries:
                    return lr * args.basic_lr_annealing.lr_decay
            else:
                raise Exception("Unspecified learning rate annealing strategy")

        elif hasattr(args, 'cosine_decay'):
            # Cosine annealing learning rate scheduler with periodic restarts.
            return tf.train.cosine_decay(lr, step, args.cosine_decay.decay_steps)

        else:
            # Don't perform any learning rate annealing
            return lr

    except AttributeError:
        print("Invalid learning rate annealing structure in config")


