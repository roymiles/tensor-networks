import numpy as np

# temp
import sys
sys.path.append('/home/roy/PycharmProjects/TensorNetworks/')

from Architectures.impl.CIFARExample import CIFARExample
from Architectures.impl.MobileNetV1 import MobileNetV1
from Architectures.impl.MobileNetV2 import MobileNetV2
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

# set random seeds
random.seed(1234)
tf.random.set_random_seed(1234)
np.random.seed(1234)

print(tf.__version__)

if __name__ == '__main__':

    num_classes = 100
    dataset_name = 'cifar100'
    batch_size = 128
    num_epochs = 60
    initial_learning_rate = 0.1
    switch_list = [0.4, 0.6, 0.8, 1.0]

    architecture = MobileNetV2(num_classes=num_classes)
    #architecture = CIFARExample(num_classes=num_classes)

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
    ds_test = ds_test.batch(2000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y = tf.placeholder(tf.float32, shape=[None, num_classes])
        # The current switch being used for inference (from switch_list)
        switch_idx = tf.placeholder(tf.int32, shape=[])

    use_tucker = False
    if use_tucker:
        model = TuckerNet(architecture=architecture)
        model.build(conv_ranks=conv_ranks, fc_ranks=fc_ranks, switch_list=switch_list,
                    name="MyTuckerNetwork")
        logits_op = model(input=x, switch_idx=switch_idx)
    else:
        print("Using StandardNet")
        model = StandardNetwork(architecture=architecture)
        model.build("MyStandardNetwork")
        logits_op = model(input=x)

        # temp for mobilenetv2
        logits_op = tf.reduce_mean(logits_op, axis=[1, 2])

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits_op)
    loss_op = tf.reduce_mean(loss_op)

    # Add the regularisation terms
    #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #_lambda = 0.01  # Scaling factor
    #loss_op += loss_op + _lambda * sum(reg_losses)

    print(y)
    print(logits_op)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_op, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               100000, 1e-6, staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9,
                                           use_nesterov=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss_op, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # Help for debugging nan gradients
    grads_and_vars = optimizer.compute_gradients(loss_op)
    # print(grads_and_vars[-4][1].name)
    # exit()

    for g, v in grads_and_vars:
        print(v.name)
        # tf.summary.histogram(v.name, v)
        # tf.summary.histogram(v.name + '_grad', g)

    # Create session and initialize weights
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    )
    sess = tf.Session(config=config)
    sess.run(init_op)

    # Tensorboard
    merged = tf.summary.merge_all()
    log_dir = conf.log_dir + dataset_name
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print("Run: \"tensorboard --logdir={}\"".format(log_dir))

    num_params = 0
    for v in tf.trainable_variables():
        num_params += tfvar_size(v)

    print("Number of parameters = {}".format(num_params))

    # for debugging
    # w = model.get_weights()

    for epoch in tqdm(range(num_epochs)):

        # Training
        for batch in tfds.as_numpy(ds_train):
            images, labels = batch['image'], batch['label']

            # Normalise in range [0, 1)
            images = images / 255.0

            # print("im: {}".format(images[0, 0, :, 0]))

            # One hot encode
            labels = np.eye(num_classes)[labels]

            # Choose a random switch at every step
            i = random.randrange(len(switch_list))
            switch = switch_list[i]

            feed_dict = {
                x: images,
                y: labels,
                switch_idx: i
            }

            fetches = [global_step, train_op, loss_op, merged]
            step, _, loss, summary = sess.run(fetches, feed_dict)

            # if step % 10 == 0:
            # train_writer.add_summary(summary, step)

            if step % 100 == 0:
                print("Epoch: {}, Step {}, Loss: {}, Switch: {}".format(epoch, step, loss, switch))

    # Testing (after training)
    for i, switch in enumerate(switch_list):
        for batch in tfds.as_numpy(ds_test):
            images, labels = batch['image'], batch['label']

            # Normalise in range [0, 1)
            images = images / 255.0

            # One hot encode
            labels = np.eye(num_classes)[labels]

            feed_dict = {
                x: images,
                y: labels,
                switch_idx: i
            }

            acc = sess.run(accuracy, feed_dict)
            print("Accuracy = {}, Switch = {}".format(acc, switch))

    exit()
    # Get the weights
    _w1 = model.get_weights().get_layer_weights(13)['kernel'].combine(reshape=["I", "O"])
    _w2 = model.get_weights().get_layer_weights(16)['kernel'].combine(reshape=["I", "O"])

    _w3 = model.get_weights().get_layer_weights(8)['kernel'].combine(reshape=["W", "H", "C", "N"])

    w1, w2, w3 = sess.run([_w1, _w2, _w3], feed_dict={})

    print(w3)
    print(w3.shape)
    visualise_volume_slices(w3[:, 0, :, :])
    print("fin")

    #print("var1_I = {}".format(np.sum(np.var(w1, axis=0))))
    #print("var1_O = {}".format(np.sum(np.var(w1, axis=1))))
    #print("var2_I = {}".format(np.sum(np.var(w2, axis=0))))
    #print("var2_O = {}".format(np.sum(np.var(w2, axis=1))))

    #cv2.imshow("Fc1", w1)
    #cv2.imshow("Fc2", w2)
    #cv2.waitKey()
