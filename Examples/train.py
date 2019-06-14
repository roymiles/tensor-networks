"""
    Generic training code for arbitrary models/datasets
    The configuration is specified by the arguments
"""

import numpy as np

# temporary fudge, so can call script from terminal
import sys
sys.path.append('/home/roy/PycharmProjects/TensorNetworks/')

import tensorflow_datasets as tfds
import os
from tqdm import tqdm
import config as conf
from base import *
from tflite import *
from Examples.config.utils import load_config, get_architecture, get_optimizer, get_data_augmentation_fn

from Networks.network import Network as MyNetwork
from transforms import random_horizontal_flip, normalize
import numpy as np


# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

if __name__ == '__main__':

    # Change if want to test different model/dataset
    args = load_config("MobileNetV2_MNIST.json")
    ds_args = load_config(f"datasets/{args.dataset_name}.json")

    if hasattr(args, 'seed'):
        seed = args.seed
    else:
        seed = 1234

    # set random seeds
    random.seed(seed)
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    architecture = get_architecture(args, ds_args)

    # See available datasets
    print(tfds.list_builders())

    # Already downloaded
    datasets = tfds.load(args.dataset_name.lower(), data_dir=conf.tfds_dir)
    info = tfds.builder(args.dataset_name.lower()).info
    label_names = info.features['label'].names

    ds_train, ds_test = datasets['train'], datasets['test']  # Uses "test" on CIFAR, MNIST

    # Build your input pipeline
    # See: https://stackoverflow.com/questions/55141076/how-to-apply-data-augmentation-in-tensorflow-2-0-after-tfds-load
    # ds_train.map(
    #     lambda image, label: (tf.image.random_flip_left_right(image), label)
    # )
    ds_train = ds_train.map(
         lambda x: {
             "image": get_data_augmentation_fn(ds_args)(x['image']),
             "label": x['label'],
             # "file_name": x['file_name']
         }
    ).shuffle(args.batch_size * 50).batch(args.batch_size)

    ds_test = ds_test.map(
        lambda x: {
            "image": tf.image.resize_image_with_crop_or_pad(x['image'], target_width=ds_args.img_width,
                                                            target_height=ds_args.img_height),
            "label": x['label'],
            # "file_name": x['file_name']
        }
    ).shuffle(args.batch_size * 50)

    train_iterator = ds_train.make_initializable_iterator()
    next_train_element = train_iterator.get_next()

    test_iterator = ds_train.make_initializable_iterator()
    next_test_element = test_iterator.get_next()

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, ds_args.img_width, ds_args.img_height, ds_args.num_channels])
        y = tf.placeholder(tf.float32, shape=[None, ds_args.num_classes])
        is_training = tf.placeholder(tf.bool, shape=[])

    model = MyNetwork(architecture=architecture, switches=[0.5, 1.0])
    model.build("MyNetwork")

    logits_op = model(x, is_training=is_training)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, logits_op))
    tf.summary.scalar('Training Loss', loss_op)

    # Add the regularisation terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_op += loss_op + sum(reg_losses)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_op, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Testing Accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    opt, learning_rate = get_optimizer(args)
    train_op = opt.minimize(loss_op, global_step=global_step)

    init_op = tf.global_variables_initializer()

    # Create session and initialize weights
    config = tf.ConfigProto(
        # device_count={'GPU': 0},  # If want to run on CPU only
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)
        
    )
    sess = tf.InteractiveSession(config=config)  # Session, Interactive session catches errors better
    sess.run(init_op)

    # Tensorboard
    merged = tf.summary.merge_all()
    log_dir = f"{conf.log_dir}/{args.dataset_name}/{args.architecture}"
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print("Run: tensorboard --logdir=\"{}\"".format(log_dir))

    num_params = 0
    for v in tf.trainable_variables():
        # print(v)
        num_params += tfvar_size(v)

    print("Number of parameters = {}".format(num_params))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    lr = args.learning_rate
    pbar = tqdm(range(args.num_epochs))
    for epoch in pbar:

        # Training
        sess.run(train_iterator.initializer)
        num_batch = 0
        while True:
            try:
                batch = sess.run(next_train_element)
                images, labels = batch['image'], batch['label']

                images = images / 255.0

                # mean = [0.485, 0.456, 0.406]
                # std = [0.229, 0.224, 0.225]
                # images = normalize(images, mean, std)

                # One hot encode
                labels = np.eye(ds_args.num_classes)[labels]

                feed_dict = {
                    x: images,
                    y: labels,
                    learning_rate: lr,
                    is_training: True
                }

                fetches = [global_step, train_op, loss_op, merged]
                step, _, loss, summary = sess.run(fetches, feed_dict)

                num_batch += 1
                if step % 100 == 0:
                    train_writer.add_summary(summary, step)

                    # Standard description
                    pbar.set_description(f"Epoch: {epoch}, Step {step}, Loss: {loss}, Learning rate: {lr}")

                    # Save the variables to disk.
                    save_path = saver.save(sess, f"{conf.ckpt_dir}/{args.dataset_name}_{args.architecture}_{step}.ckpt")

            except tf.errors.OutOfRangeError:
                # print(f"Batch num = {num_batch}")
                break

        # Decay learning rate every n epochs
        if epoch % args.num_epochs_decay == 0 and epoch != 0:
            lr = lr * args.learning_rate_decay

        if epoch % args.test_every == 0 and epoch != 0:

            # Test step
            sess.run(test_iterator.initializer)
            # acc2 = []
            while True:
                try:
                    batch = sess.run(next_test_element)
                    images, labels = batch['image'], batch['label']

                    images = images / 255.0

                    # One hot encode
                    labels = np.eye(ds_args.num_classes)[labels]

                    feed_dict = {
                        x: images,
                        y: labels,
                        is_training: False
                    }

                    fetches = [accuracy, merged]
                    acc, summary = sess.run(fetches, feed_dict)
                    # train_writer.add_summary(summary, step)

                    # TODO: PLEASE FIX THIS, WHY NOT COMPATABLE WITH TENSORBOARD SCALAR PLOT
                    # print("Test accuracy = {}".format(acc))
                    # acc2.append(acc)
                except tf.errors.OutOfRangeError:
                    break

            # print(f"Test accuracy {np.mean(acc2)}")

    # Export standard model
    save_path = f"{conf.save_dir}/{args.dataset_name}_{args.architecture}.pbtxt"
    tf.train.write_graph(sess.graph.as_graph_def(), '.', save_path, as_text=True)

    # Save the labels
    with open(f"{conf.labels_dir}/{args.dataset_name}.txt", 'w') as f:
        for item in label_names:
            f.write("%s\n" % item)

    # Freeze the graph
    frozen_graph = freeze_graph(sess, ["layer_82/BiasAdd"])  # logits_op.name

    # Export model tflite
    export_tflite_from_frozen_graph(frozen_graph, input_nodes=[x], output_nodes=[logits_op],
                                    dataset_name=args.dataset_name, architecture_name=args.architecture)

    print("Finished")

