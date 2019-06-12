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
from tflite import export_tflite_from_session
from Examples.config.utils import load_config, get_architecture, get_optimizer

from Networks.impl.standard import StandardNetwork as MyNetwork
from transforms import random_horizontal_flip, normalize
import numpy as np


# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

if __name__ == '__main__':

    # Change if want to test different model/dataset
    tc = load_config("mnist_example.json")

    if "seed" in tc.keys():
        seed = tc['seed']
    else:
        seed = 1234

    # set random seeds
    random.seed(seed)
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    architecture = get_architecture(tc['architecture'])

    # See available datasets
    print(tfds.list_builders())

    # Already downloaded
    datasets = tfds.load(tc['dataset_name'], data_dir=conf.tfds_dir)

    ds_train, ds_test = datasets['train'], datasets['test']

    # Build your input pipeline
    ds_train = ds_train.padded_batch(
        batch_size=tc['batch_size'],
        padded_shapes={
          'label': [],
          'image': [-1, -1, -1]
        }).shuffle(1000).prefetch(1000)

    # No batching just use entire test data
    ds_test = ds_test.batch(2000).shuffle(1000).prefetch(1000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, tc['img_width'], tc['img_height'], tc['num_channels']])
        y = tf.placeholder(tf.float32, shape=[None, tc['num_classes']])
        learning_rate = tf.placeholder(tf.float64, shape=[])

    model = MyNetwork(architecture=architecture)
    model.build("MyNetwork")

    logits_op = model(input=x)

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits_op)
    loss_op = tf.reduce_mean(loss_op)
    tf.summary.scalar('Training Loss', loss_op)

    # Add the regularisation terms
    #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #loss_op += loss_op + sum(reg_losses)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_op, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Testing Accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = get_optimizer(tc).minimize(loss_op, global_step=global_step)

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
    log_dir = conf.log_dir + tc['dataset_name']
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print("Run: \"tensorboard --logdir={}\"".format(log_dir))

    num_params = 0
    for v in tf.trainable_variables():
        print(v)
        num_params += tfvar_size(v)

    print("Number of parameters = {}".format(num_params))

    def preprocess_batch(images):
        #images = random_horizontal_flip(images)

        images = np.array(images) / 255.0
        #images = (images - 0.449) / 0.226
        return images

    lr = tc['learning_rate']
    pbar = tqdm(range(tc['num_epochs']))
    for epoch in pbar:

        # Training
        for batch in tfds.as_numpy(ds_train):
            images, labels = batch['image'], batch['label']

            # Pad to appropriate size (32x32x3)
            # images = np.array([uwotm8(img) for img in images])

            images = preprocess_batch(images)

            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # images = normalize(images, mean, std)

            # One hot encode
            labels = np.eye(tc['num_classes'])[labels]

            feed_dict = {
                x: images,
                y: labels,
                learning_rate: lr
            }

            fetches = [global_step, train_op, loss_op, merged, logits_op]
            step, _, loss, summary, pred = sess.run(fetches, feed_dict)

            if step % 100 == 0:
                train_writer.add_summary(summary, step)

                # Standard description
                pbar.set_description(f"Epoch: {epoch}, Step {step}, Loss: {loss}, Learning rate: {lr}")

                # See if logits have blown up
                # pbar.set_description(f"Epoch: {epoch}, Step {step}, Pred: {pred[0]}, Trg: {labels[0]}")

        # Decay learning rate every n epochs
        if epoch % tc['num_epochs_decay'] == 0 and epoch != 0:
            lr = lr * tc['learning_rate_decay']

        if epoch % tc['test_every'] == 0 and epoch != 0:
            for batch in tfds.as_numpy(ds_test):
                images, labels = batch['image'], batch['label']

                images = preprocess_batch(images)

                # One hot encode
                labels = np.eye(tc['num_classes'])[labels]

                feed_dict = {
                    x: images,
                    y: labels,
                }

                acc = sess.run(accuracy, feed_dict)
                print("Accuracy = {}".format(acc))

    # Export model tflite
    export_tflite_from_session(sess, input_nodes=[x], output_nodes=[logits_op], name="cifar")
