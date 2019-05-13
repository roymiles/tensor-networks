""" Core training script """

import numpy as np
from architectures import *
from Networks.tensor_network_v1 import TensorNetV1
from Networks.standard_network import StandardNetwork
import tensorflow_datasets as tfds
import config as conf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

if __name__ == '__main__':
    # These hyperparameters control the compression
    # of the convolutional and fully connected weights
    conv_ranks = [5, 5, 5]
    fc_ranks = [5, 5]

    # NOTE: Just change architecture here
    architecture = CIFAR100Example()

    # See available datasets
    print(tfds.list_builders())

    # Fetch the dataset directly
    data = tfds.builder(conf.tfds_dataset_name)

    # Download the data, prepare it, and write it to disk
    data.download_and_prepare(download_dir=conf.tfds_dir)

    # Load data from disk as tf.data.Datasets
    datasets = data.as_dataset()
    ds_train, ds_test = datasets['train'], datasets['test']

    # Build your input pipeline
    ds_train = ds_train.shuffle(20000).batch(conf.batch_size).prefetch(20000)
    ds_test = ds_test.shuffle(20000).batch(conf.batch_size).prefetch(20000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y = tf.placeholder(tf.float32, shape=[None, conf.num_classes])
        learning_rate = tf.placeholder(tf.float32, shape=[])

    #model_v1 = TensorNetV1(architecture=architecture,
    #                       conv_ranks=conv_ranks,
    #                       fc_ranks=fc_ranks)

    model_v2 = StandardNetwork(architecture=architecture)

    # Single forward pass
    logits = model_v2(input=x)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)
    avg_loss = tf.reduce_mean(loss)  # Over entire batch

    # ...
    predictions = tf.argmax(input=logits, axis=1)
    ground_truths = tf.argmax(input=y, axis=1)
    acc_op = tf.metrics.accuracy(ground_truths, predictions)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.train.MomentumOptimizer(learning_rate=conf.initial_learning_rate,
                                          momentum=0.9, use_nesterov=True).minimize(loss, global_step=global_step)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Initialize weights
        sess.run(init_op)

        #print("Number of parameters *model_v1* = {}".format(model_v1.num_parameters()))
        print("Number of parameters *model_v2* = {}".format(model_v2.num_parameters()))

        for epoch in range(conf.epochs):

            # Training
            for batch in tfds.as_numpy(ds_train):
                images, labels = batch['image'], batch['label']

                # One hot encode
                labels = np.eye(conf.num_classes)[labels]

                feed_dict = {
                    x: images,
                    y: labels,
                    learning_rate: conf.initial_learning_rate
                }

                fetches = [global_step, train_op, avg_loss]

                step, _, batch_loss = sess.run(fetches, feed_dict)

                # if step % 100 == 0:
                #    print("Epoch: {}, Step {}, Loss: {}".format(epoch, step, batch_loss))

            # Validation
            avg_acc = []
            for batch in tfds.as_numpy(ds_test):
                images, labels = batch['image'], batch['label']

                # One hot encode
                labels = np.eye(conf.num_classes)[labels]

                feed_dict = {
                    x: images,
                    y: labels
                }

                fetches = acc_op
                val_acc, _ = sess.run(fetches, feed_dict)
                avg_acc.append(val_acc)

            print("Epoch {}, Validation accuracy: {}".format(epoch, np.mean(avg_acc)))
