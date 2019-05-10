import tensorflow as tf
import numpy as np
from layers import *
from architectures import *
from tensor_network import TensorNetV1
from standard_network import StandardNetwork
import datasets
import tensorflow_datasets as tfds
import config as conf
from tqdm import tqdm

import os
# Suppress warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # These hyperparameters control the compression
    # of the convolutional and fully connected weights
    conv_ranks = [5, 5, 5, 5]
    fc_ranks = [3, 3, 3]

    # NOTE: Just change architecture here
    architecture = CIFAR100Example()
    print("Number of parameters *before* = {}".format(architecture.num_parameters()))

    # See available datasets
    print(tfds.list_builders())

    # Fetch the dataset directly
    mnist = tfds.builder('mnist')

    # Download the data, prepare it, and write it to disk
    mnist.download_and_prepare()

    # Load data from disk as tf.data.Datasets
    datasets = mnist.as_dataset()
    ds_train, ds_test = datasets['train'], datasets['test']

    # Build your input pipeline
    ds_train = ds_train.shuffle(1000).batch(conf.batch_size).prefetch(10)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        learning_rate = tf.placeholder(tf.float32, shape=[])

    #model = TensorNetV1(architecture=architecture,
    #                    conv_ranks=conv_ranks,
    #                    fc_ranks=fc_ranks)

    model = StandardNetwork(architecture=architecture)

    #print("Number of parameters *after* = {}".format(model.num_parameters()))

    # Single forward pass
    logits = model(x)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)
    avg_loss = tf.reduce_mean(loss)  # Over entire batch

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initialize weights
        sess.run(init_op)

        for batch in tfds.as_numpy(ds_train):
            images, labels = batch['image'], batch['label']

            # One hot encode
            labels = np.eye(10)[labels]

            feed_dict = {
                x: images,
                y: labels,
                learning_rate: conf.initial_learning_rate
            }

            fetches = [avg_loss, train_op]

            train_loss, _ = sess.run(fetches, feed_dict)

            print("Loss: {}".format(train_loss))

    """
    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains for 5 epochs.
    model.fit(x_train, y_train, batch_size=32, epochs=5)
    """
