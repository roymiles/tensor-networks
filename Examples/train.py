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
from tflite import *
from Examples.config.utils import load_config, get_architecture, get_optimizer, preprocess_images_fn

from Networks.network import Network as MyNetwork
from transforms import random_horizontal_flip, normalize
import numpy as np


# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

if __name__ == '__main__':

    # Change if want to test different model/dataset
    args = load_config("MobileNetV1_CIFAR100.json")
    ds_args = load_config(f"datasets/{args.dataset_name}.json")

    if hasattr(args, 'seed'):
        seed = args.seed
    else:
        seed = 1234

    # set random seeds
    random.seed(seed)
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    # See available datasets
    print(tfds.list_builders())

    # Already downloaded
    datasets = tfds.load(args.dataset_name.lower(), data_dir=conf.tfds_dir)
    info = tfds.builder(args.dataset_name.lower()).info
    label_names = info.features['label'].names

    ds_train, ds_test = datasets['train'], datasets['test']  # Uses "test" on CIFAR, MNIST

    # Build your input pipeline
    ds_train = ds_train.map(
         lambda x: {
             "image": preprocess_images_fn(ds_args)(x['image']),
             "label": x['label'],
             # "file_name": x['file_name']
         }
    ).shuffle(args.batch_size * 50).batch(args.batch_size)

    ds_test = ds_test.shuffle(args.batch_size * 50).batch(10000)

    train_iterator = ds_train.make_initializable_iterator()
    next_train_element = train_iterator.get_next()

    test_iterator = ds_test.make_initializable_iterator()
    next_test_element = test_iterator.get_next()

    with tf.variable_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, ds_args.img_width, ds_args.img_height, ds_args.num_channels])
        y = tf.placeholder(tf.float32, shape=[None, ds_args.num_classes])
        is_training = tf.placeholder(tf.bool, shape=[])
        switch_idx = tf.placeholder(tf.int32, shape=[])
        switch = tf.placeholder(tf.float32, shape=[])

    architecture = get_architecture(args, ds_args)
    model = MyNetwork(architecture=architecture)
    model.build("MyNetwork")

    logits_op = model(x, is_training=is_training, switch_idx=switch_idx, switch=switch)

    with tf.variable_scope("loss"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, logits_op))
        tf.summary.scalar('training_loss', loss_op, collections=['train'])
        tf.summary.scalar('testing_loss', loss_op, collections=['test'])

    with tf.variable_scope("regularisation"):
        # Add the regularisation terms
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_op += loss_op + sum(reg_losses)

    with tf.variable_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_op, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Testing Accuracy', accuracy_op, collections=['test'])

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # This update op ensures the moving averages for BN
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt, learning_rate = get_optimizer(args)
        train_op = opt.minimize(loss_op, global_step=global_step)

    init_op = tf.global_variables_initializer()

    # Create session and initialize weights
    config = tf.ConfigProto(
        # device_count={'GPU': 0},  # If want to run on CPU only
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)

    )
    sess = tf.InteractiveSession(config=config)  # Session, Interactive session catches errors better
    sess.run(init_op)

    # Tensorboard
    train_summary = tf.summary.merge_all('train')
    test_summary = tf.summary.merge_all('test')

    log_dir = f"{conf.log_dir}/{args.dataset_name}/{args.architecture}"
    write_op = tf.summary.FileWriter(log_dir, sess.graph)
    # test_writer = tf.summary.FileWriter(log_dir + "/test", sess.graph)

    print("Run: tensorboard --logdir=\"{}\"".format(log_dir))

    num_params = 0
    for v in tf.trainable_variables():
        # print(v)
        num_params += tfvar_size(v)

    print("Number of parameters = {}".format(num_params))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Make checkpoint save path if does not exist
    checkpoint_dir = f"{conf.ckpt_dir}/{args.dataset_name}/{args.architecture}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    lr = args.learning_rate
    pbar = tqdm(range(args.num_epochs))
    for epoch in pbar:

        # ---------------- TRAINING ---------------- #
        sess.run(train_iterator.initializer)
        num_batch = 0
        while True:
            try:
                batch = sess.run(next_train_element)
                images, labels = batch['image'], batch['label']

                images = images / 255.0

                # mean = [0.485, 0.456, 0.406]
                # std = [0.229, 0.224, 0.225]
                # images = normalize(images, mean, std)

                # One hot encode
                labels = np.eye(ds_args.num_classes)[labels]

                sw_idx, sw = random.choice(list(enumerate(args.switch_list)))
                feed_dict = {
                    x: images,
                    y: labels,
                    learning_rate: lr,
                    is_training: True,
                    switch_idx: sw_idx,
                    switch: sw
                }

                fetches = [global_step, train_op, loss_op, train_summary, logits_op]
                step, _, loss, summary, logits = sess.run(fetches, feed_dict)

                num_batch += 1
                if step % 100 == 0:
                    write_op.add_summary(summary, step)

                    # Standard description
                    pbar.set_description(f"Epoch: {epoch}, Step {step}, Loss: {loss}, Learning rate: {lr}")
                    # pbar.set_description(f"Epoch: {epoch}, Step {step}, Logits: {logits}")

                    # Save the variables to disk.
                    save_path = saver.save(sess, f"{checkpoint_dir}/{step}.ckpt")

            except tf.errors.OutOfRangeError:
                # print(f"Batch num = {num_batch}, Num images seen = {num_batch * args.batch_size}")
                break

        # Decay learning rate every n epochs
        if epoch % args.num_epochs_decay == 0 and epoch != 0:
            lr = lr * args.learning_rate_decay

        # ---------------- TESTING ---------------- #
        if epoch % args.test_every == 0:

            sess.run(test_iterator.initializer)
            # Check results on all switches
            for sw_idx, sw in enumerate(args.switch_list):
                while True:
                    try:
                        batch = sess.run(next_test_element)
                        images, labels = batch['image'], batch['label']

                        images = images / 255.0

                        # One hot encode
                        labels = np.eye(ds_args.num_classes)[labels]

                        feed_dict = {
                            x: images,
                            y: labels,
                            is_training: False,
                            switch_idx: sw_idx,
                            switch: sw
                        }

                        fetches = [accuracy_op, loss_op, test_summary, logits_op]
                        _, _, summary, logits = sess.run(fetches, feed_dict)
                        write_op.add_summary(summary, step)
                        # print(f"Testing Logits: {logits}")

                        # print("Test accuracy = {}".format(acc))
                    except tf.errors.OutOfRangeError:
                        break

    # Export standard model
    save_path = f"{conf.save_dir}/{args.dataset_name}_{args.architecture}.pbtxt"
    tf.train.write_graph(sess.graph.as_graph_def(), '.', save_path, as_text=True)

    # Save the labels
    with open(f"{conf.labels_dir}/{args.dataset_name}.txt", 'w') as f:
        for item in label_names:
            f.write("%s\n" % item)

    # Freeze the graph
    frozen_graph = freeze_graph(sess, ["layer_82/BiasAdd"])  # logits_op.name

    # Export model tflite
    export_tflite_from_frozen_graph(frozen_graph, input_nodes=[x], output_nodes=[logits_op],
                                    dataset_name=args.dataset_name, architecture_name=args.architecture)

    print("Finished")

