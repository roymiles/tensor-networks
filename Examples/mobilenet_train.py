import numpy as np

from Architectures.impl.MobileNetV1 import MobileNetV1
from Networks.impl.standard import StandardNetwork

import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import config as conf

# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

if __name__ == '__main__':
    architecture = MobileNetV1()

    # See available datasets
    print(tfds.list_builders())

    # Fetch the dataset directly
    mnist = tfds.builder('imagenet2012')

    # Download the data, prepare it, and write it to disk
    mnist.download_and_prepare(download_dir=conf.tfds_dir)

    # Load data from disk as tf.data.Datasets
    datasets = mnist.as_dataset()
    ds_train, ds_test = datasets['train'], datasets['test']

    # Build your input pipeline
    batch_size = 124
    ds_train = ds_train.batch(batch_size).prefetch(10)

    # No batching just use entire test data
    ds_test = ds_test.batch(10000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        learning_rate = tf.placeholder(tf.float32, shape=[])

    model = StandardNetwork(architecture=architecture)
    model.build("MyStandardMobileNetV1")

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

    # Create session and initialize weights
    sess = tf.Session()
    sess.run(init_op)

    # Tensorboard
    merged = tf.summary.merge_all()
    log_dir = '/home/roy/Desktop/Tensorboard/MobileNetV1'
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print("Run: \"tensorboard --logdir={}\"".format(log_dir))

    print("Number of parameters = {}".format(model.num_parameters()))

    num_epochs = 12
    initial_learning_rate = 1e-4

    for epoch in tqdm(range(num_epochs)):

        # Training
        for batch in tfds.as_numpy(ds_train):
            images, labels = batch['image'], batch['label']

            # For debugging, show first image
            cv2.imshow("image", images[0])
            cv2.waitKey()

            # Normalise in range [0, 1)
            images = images / 255.0

            # One hot encode
            labels = np.eye(10)[labels]

            feed_dict = {
                x: images,
                y: labels,
                learning_rate: initial_learning_rate
            }

            fetches = [global_step, train_op, avg_loss]

            step, _,  loss = sess.run(fetches, feed_dict)

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