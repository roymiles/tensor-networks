import numpy as np

from Architectures.impl.MNISTExample import MNISTExample
from Networks.impl.standard import StandardNetwork
from Networks.impl.tucker_like import TuckerNet

import tensorflow_datasets as tfds
import config as conf
import tensorflow as tf
import cv2

print(tf.__version__)

if __name__ == '__main__':
    # NOTE: Just change architecture here
    architecture = MNISTExample()

    # These hyperparameters control the compression
    # of the convolutional and fully connected weights
    conv_ranks = {
        0: [5, 5, 5],
        2: [5, 5, 5],
        6: [5, 5, 5],
        8: [5, 5, 5]
    }
    fc_ranks = {
        13: [5, 5],
        16: [5, 5]
    }

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

    # No batching just use entire test data
    ds_test = ds_test.batch(10000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        learning_rate = tf.placeholder(tf.float32, shape=[])

    # model_v1 = TuckerNet(architecture=architecture)
    # model_v1.build(conv_ranks=conv_ranks, fc_ranks=fc_ranks, name="MyTuckerNetwork")

    model = StandardNetwork(architecture=architecture)
    model.build("MyStandardNetwork")

    # Single forward pass
    logits = model(input=x)

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)
    avg_loss = tf.reduce_mean(loss_op)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, global_step=global_step)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/home/roy/Desktop/Tensorboard', sess.graph)
        print("Run: \"tensorboard --logdir=/home/roy/Desktop/Tensorboard\"")

        # Initialize weights
        sess.run(init_op)

        print("Number of parameters = {}".format(model.num_parameters()))

        for epoch in range(conf.epochs):

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

                feed_dict = {
                    x: images,
                    y: labels,
                    learning_rate: conf.initial_learning_rate
                }

                fetches = [global_step, train_op, avg_loss]

                step, _,  loss = sess.run(fetches, feed_dict)

                # if step % 100 == 0:
                #    print("Epoch: {}, Step {}, Loss: {}".format(epoch, step, loss))

        # Testing
        for batch in tfds.as_numpy(ds_test):
            images, labels = batch['image'], batch['label']

            # Normalise in range [0, 1)
            images = images / 255.0

            print(images.shape)

            # One hot encode
            labels = np.eye(10)[labels]

            feed_dict = {
                x: images,
                y: labels
            }

            acc = sess.run(accuracy, feed_dict)
            print("Accuracy = {}".format(acc))
