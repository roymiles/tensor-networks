import numpy as np

# temp
import sys
sys.path.append('/home/roy/PycharmProjects/TensorNetworks/')

from Architectures.impl.DenseNet import TransitionLayer, DenseBlock
from Networks.impl.sandbox import TuckerNet
from Networks.impl.standard import StandardNetwork

import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import config as conf
from base import *
from tflite import export_tflite_from_session


# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# set random seeds
random.seed(1234)
tf.random.set_random_seed(1234)
np.random.seed(1234)

print(tf.__version__)

if __name__ == '__main__':

    num_classes = 10
    dataset_name = 'cifar10'
    batch_size = 64
    num_epochs = 12
    switch_list = [1.0]

    # Already downloaded
    datasets = tfds.load(dataset_name, data_dir=conf.tfds_dir)

    ds_train, ds_test = datasets['train'], datasets['test']

    # Build your input pipeline
    ds_train = ds_train.batch(batch_size).prefetch(1000)

    # No batching just use entire test data
    ds_test = ds_test.batch(2000)

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y = tf.placeholder(tf.float32, shape=[None, num_classes])

    # ----- CREATE THE DENSE NET ----- #

    # Growth rate
    k = 12
    net = tf.keras.layers.BatchNormalization()(x)
    net = tf.nn.relu(net)
    net = tf.keras.layers.Conv2D(filters=k, kernel_size=(7, 7))(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(net)

    print("in {}".format(net))
    d1 = DenseBlock(n=6,   # Number of layers
                    k0=k,  # Input dims
                    k=k)   # Growth rate
    model1 = StandardNetwork(architecture=d1)
    model1.build("DenseBlock1")
    net = model1(input=net)
    c = d1.get_output_dim()
    print("out {}".format(net))

    t1 = TransitionLayer(k0=c)
    model2 = StandardNetwork(architecture=t1)
    model2.build("TransitionLayer1")
    net = model2(input=net)
    # Does not affect output dims

    d2 = DenseBlock(n=12, k0=c, k=k)
    model3 = StandardNetwork(architecture=d2)
    model3.build("DenseBlock2")
    net = model3(input=net)
    c = d2.get_output_dim()

    t2 = TransitionLayer(k0=c)
    model4 = StandardNetwork(architecture=t2)
    model4.build("TransitionLayer2")
    net = model4(input=net)

    d3 = DenseBlock(n=32, k0=c, k=k)
    model5 = StandardNetwork(architecture=d3)
    model5.build("DenseBlock3")
    net = model5(input=net)
    c = d3.get_output_dim()

    t3 = TransitionLayer(k0=c)
    model6 = StandardNetwork(architecture=t3)
    model6.build("TransitionLayer2")
    net = model6(input=net)

    # Global average pool
    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

    # 1000D Fully connected layer
    logits_op = tf.contrib.layers.fully_connected(net, 1000)

    # ----- FIN ----- #

    num_params = 0
    for v in tf.trainable_variables():
        print(v)
        num_params += tfvar_size(v)

    print("Number of parameters = {}".format(num_params))
    exit()

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits_op)
    loss_op = tf.reduce_mean(loss_op)

    # Add the regularisation terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_op += loss_op + sum(reg_losses)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_op, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = training_params["optimizer"].minimize(loss_op, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # Help for debugging nan gradients
    grads_and_vars = training_params["optimizer"].compute_gradients(loss_op)
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

    # Export model tflite
    export_tflite_from_session(sess, input_nodes=[x], output_nodes=[logits_op], name="cifar")

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