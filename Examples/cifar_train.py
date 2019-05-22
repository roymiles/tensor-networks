import numpy as np

# temp
import sys
sys.path.append('/home/roy/PycharmProjects/TensorNetworks/')

from Architectures.impl.CIFARExample import CIFARExample
from Networks.impl.tucker_like import TuckerNet
from Networks.impl.standard import StandardNetwork

import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import config as conf
from base import *


# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

if __name__ == '__main__':

    num_classes = 10
    dataset_name = 'cifar10'
    batch_size = 128
    num_epochs = 12
    initial_learning_rate = 0.001

    architecture = CIFARExample(num_classes=num_classes)

    # These hyperparameters control the compression
    # of the convolutional and fully connected weights
    conv_ranks = {
        0: [6, 8, 16],
        3: [6, 16, 16],
        8: [6, 16, 32],
        11: [6, 16, 16]
    }
    fc_ranks = {
        17: [2056, 1024],
        20: [256, 128]
    }

    # See available datasets
    print(tfds.list_builders())

    # Fetch the dataset directly
    cifar = tfds.builder(dataset_name)

    # Download the data, prepare it, and write it to disk
    cifar.download_and_prepare(download_dir=conf.tfds_dir)

    # Load data from disk as tf.data.Datasets
    datasets = cifar.as_dataset()
    ds_train, ds_test = datasets['train'], datasets['test']

    # Build your input pipeline
    ds_train = ds_train.batch(batch_size).prefetch(1000)

    # No batching just use entire test data
    ds_test = ds_test.batch(10000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y = tf.placeholder(tf.float32, shape=[None, num_classes])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        # The current switch being used for inference (from switch_list)
        switch_idx = tf.placeholder(tf.int32, shape=[])

    use_tucker = False
    if use_tucker:
        model = TuckerNet(architecture=architecture)
        model.build(conv_ranks=conv_ranks, fc_ranks=fc_ranks, switch_list=[0.1, 0.4, 0.5, 0.8, 1.0],
                    name="MyTuckerNetwork")
        logits = model(input=x, switch_idx=switch_idx)
    else:
        print("Using StandardNet")
        model = StandardNetwork(architecture=architecture)
        model.build("MyStandardNetwork")
        logits = model(input=x)

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)
    loss_op = tf.reduce_mean(loss_op)

    # Add the regularisation terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    _lambda = 0.01  # Scaling factor
    loss_op += loss_op + _lambda * sum(reg_losses)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    # train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate,
    #                                      momentum=0.9,
    #                                      use_nesterov=True).minimize(loss_op, global_step=global_step)
    train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_op, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # Create session and initialize weights
    config = tf.ConfigProto(
        # device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)
    sess.run(init_op)

    # Tensorboard
    merged = tf.summary.merge_all()
    log_dir = conf.log_dir + dataset_name
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print("Run: \"tensorboard --logdir={}\"".format(log_dir))

    print("Number of parameters = {}".format(model.num_parameters()))

    # for debugging
    w = model.get_weights()
    w0 = tf.reduce_mean(w.get_layer_weights(layer_idx=0)["kernel"])
    w3 = tf.reduce_mean(w.get_layer_weights(layer_idx=3)["kernel"])
    w8 = tf.reduce_mean(w.get_layer_weights(layer_idx=8)["kernel"])
    w11 = tf.reduce_mean(w.get_layer_weights(layer_idx=11)["kernel"])

    for epoch in tqdm(range(num_epochs)):

        # Training
        for batch in tfds.as_numpy(ds_train):
            images, labels = batch['image'], batch['label']

            # Normalise in range [0, 1)
            images = images / 255.0

            # print("im: {}".format(images[0, 0, :, 0]))

            # One hot encode
            labels = np.eye(num_classes)[labels]

            feed_dict = {
                x: images,
                y: labels,
                learning_rate: initial_learning_rate,
                switch_idx: 4
            }

            fetches = [global_step, train_op, loss_op, merged, w0, w3, w8, w11]

            step, _, loss, summary, ww0, ww3, ww8, ww11 = sess.run(fetches, feed_dict)

            print("ww0: {}".format(ww0))
            print("ww3: {}".format(ww3))
            print("ww8: {}".format(ww8))
            print("ww11: {}".format(ww11))

            #if step % 10 == 0:
            train_writer.add_summary(summary, step)

            if step % 100 == 0:
                print("Epoch: {}, Step {}, Loss: {}".format(epoch, step, loss))

    # Testing (after training)
    for batch in tfds.as_numpy(ds_test):
        images, labels = batch['image'], batch['label']

        # Normalise in range [0, 1)
        images = images / 255.0

        # One hot encode
        labels = np.eye(num_classes)[labels]

        feed_dict = {
            x: images,
            y: labels,
            switch_idx: 4
        }

        acc = sess.run(accuracy, feed_dict)
        print("Accuracy = {}".format(acc))
