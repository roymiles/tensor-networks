import numpy as np

# temp
import sys
sys.path.append('/home/roy/PycharmProjects/TensorNetworks/')

from Architectures.impl.MNISTExample import MNISTExample
from Networks.impl.standard import StandardNetwork
from Networks.impl.sandbox import TuckerNet

import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import config as conf
import random


# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

if __name__ == '__main__':
    # NOTE: Just change architecture here
    architecture = MNISTExample()

    dataset_name = 'mnist'
    batch_size = 128
    switch_list = [0.1, 0.4, 0.5, 0.8, 1.0]
    num_epochs = 12
    initial_learning_rate = 0.01

    # These hyperparameters control the compression
    # of the convolutional and fully connected weights
    conv_ranks = {}
    fc_ranks = {
        1: [8, 16],
        4: [16, 8]
    }

    # See available datasets
    print(tfds.list_builders())

    # Fetch the dataset directly
    mnist = tfds.builder(dataset_name)

    # Download the data, prepare it, and write it to disk
    mnist.download_and_prepare(download_dir=conf.tfds_dir)

    # Load data from disk as tf.data.Datasets
    datasets = mnist.as_dataset()
    ds_train, ds_test = datasets['train'], datasets['test']

    # Build your input pipeline
    ds_train = ds_train.batch(batch_size).prefetch(10)

    # No batching, just use entire test data
    ds_test = ds_test.batch(10000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        learning_rate = tf.placeholder(tf.float32, shape=[])
        # The current switch being used for inference
        switch_idx = tf.placeholder(tf.int32, shape=[])

    use_tucker = True
    if use_tucker:
        print("Using TuckerNet")
        model = TuckerNet(architecture=architecture)
        model.build(conv_ranks=conv_ranks, fc_ranks=fc_ranks, switch_list=switch_list, name="MyTuckerNetwork")
        logits = model(input=x, switch_idx=switch_idx)
    else:
        print("Using StandardNet")
        model = StandardNetwork(architecture=architecture)
        model.build("MyStandardNetwork")
        logits = model(input=x)

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)
    avg_loss = tf.reduce_mean(loss_op)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # Create session and initialize weights
    sess = tf.Session()
    sess.run(init_op)

    # Tensorboard
    merged = tf.summary.merge_all()
    log_dir = conf.log_dir + dataset_name
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print("Run: \"tensorboard --logdir={}\"".format(log_dir))

    print("Number of parameters = {}".format(model.num_parameters()))

    for epoch in tqdm(range(num_epochs)):

        # Training
        for batch in tfds.as_numpy(ds_train):
            images, labels = batch['image'], batch['label']

            # For debugging, show first image
            # cv2.imshow("image", images[0])
            # cv2.waitKey()

            # Normalise in range [0, 1)
            images = images / 255.0

            # One hot encode
            labels = np.eye(10)[labels]

            # Choose a random switch at every step
            i = random.randrange(len(switch_list))
            switch = switch_list[i]

            feed_dict = {
                x: images,
                y: labels,
                learning_rate: initial_learning_rate,
                # Train batch on a random switch
                switch_idx: i
            }

            fetches = [global_step, train_op, avg_loss]

            step, _,  loss = sess.run(fetches, feed_dict)

            if step % 100 == 0:
                print("Epoch: {}, Step {}, Loss: {}, Switch: {}".format(epoch, step, loss, switch))

    # Testing (after training)
    # Test all switches
    for i, switch in enumerate(switch_list):
        for batch in tfds.as_numpy(ds_test):
            images, labels = batch['image'], batch['label']

            # Normalise in range [0, 1)
            images = images / 255.0

            print(images.shape)

            # One hot encode
            labels = np.eye(10)[labels]

            feed_dict = {
                x: images,
                y: labels,
                switch_idx: i
            }

            acc = sess.run(accuracy, feed_dict)
            print("Accuracy (switch {}) = {}".format(switch, acc))

    # Visualise the weights
    _w1 = model.get_weights().get_layer_weights(1)['kernel'].combine(reshape=["I", "O"])
    _w2 = model.get_weights().get_layer_weights(4)['kernel'].combine(reshape=["I", "O"])

    w1, w2 = sess.run([_w1, _w2], feed_dict={})

    cv2.imshow("Conv1", w1)
    cv2.imshow("Conv2", w2)
    cv2.waitKey()

