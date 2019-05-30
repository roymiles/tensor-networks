import numpy as np

# temp
import sys
sys.path.append('/home/roy/PycharmProjects/TensorNetworks/')

from Architectures.impl.MobileNetV1 import MobileNetV1
from Architectures.impl.MobileNetV2 import MobileNetV2, fc_ranks, conv_ranks
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

    # Currently all set for MobileNetV2

    num_classes = 1000
    dataset_name = 'imagenet2012'
    batch_size = 96
    num_epochs = 60
    initial_learning_rate = 0.045
    switch_list = [0.4, 0.6, 0.8, 1.0]
    decay = 0.9
    momentum = 0.9
    # Weight decay
    wd = 0.00004

    # Image width and heights
    width = 224
    height = 224

    architecture = MobileNetV2(num_classes=num_classes)
    architecture.print()

    # See available datasets
    print(tfds.list_builders())

    # Fetch the dataset directly
    # imagenet = tfds.builder(dataset_name)

    # Download the data, prepare it, and write it to disk
    # imagenet.download_and_prepare(download_dir=conf.tfds_dir, download_config=dl_config)

    # Load data from disk as tf.data.Datasets
    # datasets = imagenet.as_dataset()

    # Already downloaded
    datasets = tfds.load(dataset_name, data_dir=conf.tfds_dir)

    ds_train, ds_test = datasets['train'], datasets['validation']

    # Needed to fix inconsistent spatial sizes of images in a batch
    def preprocess_data(data):
        data['image'] = tf.image.resize_image_with_crop_or_pad(data['image'], width, height)
        return data

    # Build your input pipeline
    ds_train = ds_train.map(preprocess_data).batch(batch_size).prefetch(1000)
    print(ds_train)

    # No batching just use entire test data
    ds_test = ds_test.map(preprocess_data).batch(batch_size).prefetch(1000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[batch_size, width, height, 3])
        y = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
        # The current switch being used for inference (from switch_list)
        switch_idx = tf.placeholder(tf.int32, shape=[])
        learning_rate = tf.placeholder(tf.float32, shape=[])

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

    print(y)
    print(logits_op)

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits_op)
    loss_op = tf.reduce_mean(loss_op)

    # Weight decay using collections
    weights_norm = tf.reduce_sum(
        input_tensor=wd * tf.stack(
            [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
        ),
        name='weights_norm'
    )

    # Add the regularisation term
    loss_op += weights_norm

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_op, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum)
    train_op = optimizer.minimize(loss_op, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # Create session and initialize weights
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
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

    # model.num_parameters()
    print("Number of parameters = {}".format(num_params))

    lr = initial_learning_rate

    for epoch in tqdm(range(num_epochs)):

        # Training
        for batch in tfds.as_numpy(ds_train):
            images, labels = batch['image'], batch['label']

            # Normalise in range [0, 1)
            images = images / 255.0

            # One hot encode
            labels = np.eye(num_classes)[labels]

            # Choose a random switch at every step
            i = random.randrange(len(switch_list))
            switch = switch_list[i]

            feed_dict = {
                x: images,
                y: labels,
                switch_idx: i,
                learning_rate: lr
            }

            fetches = [global_step, train_op, loss_op, merged]
            step, _, loss, summary = sess.run(fetches, feed_dict)

            if step % 100 == 0:
                print("Epoch: {}, Step {}, Loss: {}, Switch: {}, lr: {}".format(epoch, step, loss, switch, lr))

        # Learning rate decay of 0.98 per epoch
        lr = lr * 0.98

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
