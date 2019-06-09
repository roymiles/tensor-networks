import numpy as np

# temp
import sys
sys.path.append('/home/roy/PycharmProjects/TensorNetworks/')

#from Architectures.impl.MobileNetV1 import MobileNetV1, training_params
from Architectures.impl.MobileNetV2 import MobileNetV2, training_params
from Networks.impl.sandbox import TuckerNet
from Networks.impl.standard import StandardNetwork

import tensorflow_datasets as tfds
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
    batch_size = training_params["batch_size"]
    num_epochs = 300
    switch_list = [1.0]

    architecture = MobileNetV2(num_classes=num_classes)
    # architecture = CIFARExample(num_classes=num_classes)

    # See available datasets
    print(tfds.list_builders())

    # Fetch the dataset directly
    # cifar = tfds.builder(dataset_name)

    # Download the data, prepare it, and write it to disk
    # cifar.download_and_prepare(download_dir=conf.tfds_dir, download_config=dl_config)

    # Load data from disk as tf.data.Datasets
    # datasets = cifar.as_dataset()

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
        # The current switch being used for inference (from switch_list)
        switch_idx = tf.placeholder(tf.int32, shape=[])

    use_tucker = False
    if use_tucker:
        model = TuckerNet(architecture=architecture)
        model.build(conv_ranks=training_params["conv_ranks"], fc_ranks=training_params["fc_ranks"],
                    switch_list=switch_list, name="MyTuckerNetwork")
        logits_op = model(input=x, switch_idx=switch_idx)
    else:
        print("Using StandardNet")
        model = StandardNetwork(architecture=architecture)
        model.build("MyStandardNetwork")
        logits_op = model(input=x)

    # Because before ? x 1 x 1 x ?
    logits_op = logits_op[:, 0, 0, :]

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits_op)
    loss_op = tf.reduce_mean(loss_op)
    tf.summary.scalar('Training Loss', loss_op)

    # Add the regularisation terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_op += loss_op + sum(reg_losses)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_op, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Testing Accuracy', accuracy)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = training_params["optimizer"].minimize(loss_op, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # Help for debugging nan gradients
    grads_and_vars = training_params["optimizer"].compute_gradients(loss_op)

    # Create session and initialize weights
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    )
    sess = tf.Session(config=config)
    sess.run(init_op)

    # Tensorboard
    merged = tf.summary.merge_all()
    log_dir = conf.log_dir + dataset_name + "v2"
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print("Run: \"tensorboard --logdir={}\"".format(log_dir))

    num_params = 0
    for v in tf.trainable_variables():
        print(v)
        num_params += tfvar_size(v)

    print("Number of parameters = {}".format(num_params))

    lr = training_params["initial_learning_rate"]
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
                training_params["learning_rate"]: lr
            }

            fetches = [global_step, train_op, loss_op, merged]
            step, _, loss, summary = sess.run(fetches, feed_dict)

            if step % 100 == 0:
                train_writer.add_summary(summary, step)

            # if step % 100 == 0:
            #     print(f"Epoch: {epoch}, Step {step}, Loss: {loss}, Switch: {switch}, Learning rate: {lr}")

        # Decay learning rate every epoch
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

    # Export model tflite
    export_tflite_from_session(sess, input_nodes=[x], output_nodes=[logits_op], name="cifar")