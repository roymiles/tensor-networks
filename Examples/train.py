"""
    Generic training code for arbitrary models/datasets
    The configuration is specified by the arguments
"""

import logging

# temporary fudge, so can call script from terminal
import sys
sys.path.append('..')

import tensorflow_datasets as tfds
import os
from tqdm import tqdm
import config as conf
from base import *
from tflite import *
from Examples.config.utils import *

from Networks.network import Network as MyNetwork
import numpy as np
from tensorflow.python.client import timeline


# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

if __name__ == '__main__':

    # Change if want to test different model/dataset
    args = load_config("MobileNetV1_CIFAR100.json")
    ds_args = load_config(f"datasets/{args.dataset_name}.json")

    # This is needed, else the logging file is not made (in PyCharm)
    # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if hasattr(args, 'seed'):
        seed = args.seed
    else:
        seed = 345

    # Unique name for this model and training method
    unique_name = f"arch_{args.architecture}_ds_{args.dataset_name}_opt_{args.optimizer}_seed_{seed}"
    if hasattr(args, 'method'):
        unique_name += f"_method_{args.method}"

    if hasattr(args, 'ranks'):
        unique_name += "_ranks_"
        unique_name += '_'.join(str(x) for x in args.ranks)

    if hasattr(args, 'partitions'):
        unique_name += "_partitions_"
        unique_name += '_'.join(str(x) for x in args.partitions)

    logging.basicConfig(filename=f'{conf.log_dir}/{unique_name}.log',
                        filemode='a',  # Append rather than overwrite
                        level=logging.NOTSET,  # Minimum level
                        format='%(levelname)s: %(message)s')

    # Set random seeds
    random.seed(seed)
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    # See available datasets
    print(tfds.list_builders())

    # Already downloaded
    datasets = tfds.load(args.dataset_name.lower(), data_dir=conf.tfds_dir)
    info = tfds.builder(args.dataset_name.lower()).info
    label_names = info.features['label'].names

    # Uses "test" on CIFAR, MNIST, "validation" on ImageNet
    ds_train, ds_test = datasets['train'], datasets['test']

    # Build your input pipeline
    ds_train = ds_train.map(
        lambda x: {
            "image": preprocess_images_fn(ds_args)(x['image']),
            "label": x['label'],
        }
    ).shuffle(args.batch_size * 50).batch(args.batch_size)
    # steps_per_epoch = ds_args.size / args.batch_size

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
    model.build()

    logits_op = model(x, is_training=is_training, switch_idx=switch_idx, switch=switch)

    with tf.variable_scope("loss"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, logits_op))

    with tf.variable_scope("regularisation"):
        # Add the regularisation terms
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_op += loss_op + sum(reg_losses)

    with tf.variable_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_op, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # This update op ensures the moving averages for BN
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt, learning_rate = get_optimizer(args)
        train_op = opt.minimize(loss_op, global_step=global_step)

    # Tensorboard
    train_summary = tf.summary.merge_all('train')
    test_summary = tf.summary.merge_all('test')

    num_params = 0
    for v in tf.trainable_variables():
        logging.info(v)
        num_params += tfvar_size(v)

    logging.info(f"Number of parameters = {num_params}")

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Make checkpoint save path if does not exist
    checkpoint_dir = f"{conf.checkpoint_dir}/{unique_name}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create the session and initialize the weights
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto(
        # device_count={'GPU': 0},  # If want to run on CPU only
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)
    )
    # Session, Interactive session (InteractiveSession)) catches errors better
    sess = tf.Session(config=config)
    sess.run(init_op)

    # Tensorboard file writers
    log_dir = f"{conf.tensorboard_dir}/{unique_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    write_op = tf.summary.FileWriter(log_dir, sess.graph)
    print(f"Run: tensorboard --logdir=\"{log_dir}\"")

    lr = args.learning_rate
    pbar = tqdm(range(args.num_epochs))
    step = 0
    for epoch in pbar:

        # ---------------- TRAINING ---------------- #
        sess.run(train_iterator.initializer)
        num_batch = 0
        train_loss = []
        train_acc = []
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

                fetches = [global_step, train_op, accuracy_op, loss_op, train_summary, logits_op]
                step, _, acc_, loss_, summary, logits = sess.run(fetches, feed_dict)

                train_acc.append(acc_)
                train_loss.append(loss_)

                num_batch += 1
                if step % 100 == 0:

                    # Training loss and accuracy logged every n steps (as opposed to every epoch)
                    avg_loss = np.mean(np.array(train_loss))
                    avg_acc = np.mean(np.array(train_acc))
                    # Reset loss and accuracy
                    train_loss = []
                    train_acc = []
                    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=avg_loss),
                                                     tf.Summary.Value(tag='train_accuracy', simple_value=avg_acc)])
                    write_op.add_summary(loss_summary, global_step=step)
                    write_op.add_summary(summary, global_step=step)
                    write_op.flush()

                    # Standard description
                    pbar.set_description(f"Epoch: {epoch}, Step {step}, Loss: {avg_loss}, Learning rate: {lr}")
                    # pbar.set_description(f"Epoch: {epoch}, Step {step}, Logits: {logits}")

                    # Save the variables to disk.
                    save_path = saver.save(sess, f"{checkpoint_dir}/{step}.ckpt")

                    # Log to file
                    logging.info(f"TRAIN epoch: {epoch}/{args.num_epochs}, step {step}, train_loss: {avg_loss}, "
                                 f"train_acc: {avg_acc}, lr: {lr}")

            except tf.errors.OutOfRangeError:
                # print(f"Batch num = {num_batch}, Num images seen = {num_batch * args.batch_size}")
                break

        # Auto detect problems and generate advice.
        # profiler.advise(options=opts)

        # Decay learning rate every n epochs
        if is_epoch_decay(epoch, args):
            lr = lr * args.learning_rate_decay

        # ---------------- TESTING ---------------- #
        if epoch % args.test_every == 0:

            sess.run(test_iterator.initializer)
            # Check results on all switches
            for sw_idx, sw in enumerate(args.switch_list):
                test_loss = []
                test_acc = []
                i = 0
                while True:
                    try:
                        # Do profiling during test stage
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
                        if epoch == 0:
                            # Only do the profiling once
                            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            acc_, loss_, summary, logits = sess.run(fetches, feed_dict, options=options,
                                                                    run_metadata=run_metadata)
                        else:
                            acc_, loss_, summary, logits = sess.run(fetches, feed_dict)

                        test_acc.append(acc_)
                        test_loss.append(loss_)
                        # print(f"Testing Logits: {logits}")

                        # print("Test accuracy = {}".format(acc))
                        i += 1

                    except tf.errors.OutOfRangeError:
                        break

                avg_loss = np.mean(np.array(test_loss))
                avg_acc = np.mean(np.array(test_acc))

                summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=avg_loss),
                                            tf.Summary.Value(tag='test_accuracy', simple_value=avg_acc)])

                # Log to file
                logging.info(f"TEST epoch: {epoch}/{args.num_epochs}, test_loss: {avg_loss}, test_acc: {avg_acc}")

                write_op.add_summary(summary, global_step=epoch)
                write_op.flush()

                # Create the Timeline object, and write it to a json file
                if epoch == 0:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(f'{conf.profiling_dir}/{unique_name}.json', 'w') as f:
                        f.write(chrome_trace)

    # Export standard model
    save_path = f"{conf.save_dir}/{unique_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tf.train.write_graph(sess.graph.as_graph_def(), '.', f"{save_path}/model.pbtxt", as_text=True)

    # Save the labels
    with open(f"{conf.labels_dir}/{args.dataset_name}.txt", 'w') as f:
        for item in label_names:
            f.write("%s\n" % item)

    # Freeze the graph
    # TODO: Remove hardcoded node name
    frozen_graph = freeze_graph(sess, ["layer_82/BiasAdd"])  # logits_op.name

    # Export model tflite
    export_tflite_from_frozen_graph(frozen_graph, input_nodes=[x], output_nodes=[logits_op], unique_name=unique_name)

    logging.info("Finished")

