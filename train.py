import numpy as np
from architectures import *
from Networks.tensor_network import TensorNetV1
from Networks.standard_network import StandardNetwork
import tensorflow_datasets as tfds
import config as conf

print(tf.__version__)

#import os
# Suppress warning messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # These hyperparameters control the compression
    # of the convolutional and fully connected weights
    conv_ranks = [20, 20, 20, 20]
    fc_ranks = [20, 20, 20]

    # NOTE: Just change architecture here
    architecture = CIFAR100Example()

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

    model_v1 = TensorNetV1(architecture=architecture,
                           conv_ranks=conv_ranks,
                           fc_ranks=fc_ranks)

    model_v2 = StandardNetwork(architecture=architecture)

    # Single forward pass
    logits = model_v2(x)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)
    avg_loss = tf.reduce_mean(loss)  # Over entire batch

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initialize weights
        sess.run(init_op)

        #acc_op = tf.metrics.accuracy(y, logits)

        print("Number of parameters *model_v1* = {}".format(model_v1.num_parameters()))
        print("Number of parameters *model_v2* = {}".format(model_v2.num_parameters()))

        for epoch in range(conf.epochs):

            for batch in tfds.as_numpy(ds_train):
                images, labels = batch['image'], batch['label']

                # One hot encode
                labels = np.eye(10)[labels]

                feed_dict = {
                    x: images,
                    y: labels,
                    learning_rate: conf.initial_learning_rate
                }

                fetches = [global_step, train_op]

                step, _ = sess.run(fetches, feed_dict)

                #if step % 100 == 0:
                #    print("Epoch: {}, Step {}, Acc: {}".format(epoch, step, train_acc))
