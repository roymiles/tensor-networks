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
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from tensorflow.python.platform import gfile
from transforms import normalize_images

# For knowledge distillation
from keras.applications.densenet import DenseNet169
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

# The info messages are getting tedious now
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

pipeline = [
    # "pipeline/CustomBottleNeck_64x64_0.2_0.5.json",
    "DenseNet_CIFAR100.json",
    # "pipeline/CustomBottleNeck_64x64_0.2_0.8.json",
    # "pipeline/CustomBottleNeck_64x64_0.2_1.0.json",
    # "pipeline/CustomBottleNeck_64x64_0.5_0.5.json",
    # "pipeline/CustomBottleNeck_64x64_0.8_0.5.json",
    # "pipeline/CustomBottleNeck_128x128_0.2_0.8.json",
    # "pipeline/CustomBottleNeck_128x128_0.8_0.2.json"
    # "pipeline/MobileNetV2/MobileNetV2_CIFAR10.json",
    # "pipeline/MobileNetV2/MobileNetV2_CIFAR10_1.2x0.2x0.1.2.json",
]
# Run on multiple models/architectures/learning methods

if __name__ == '__main__':

    for args_name in pipeline:
        tf.reset_default_graph()
        args = load_config(args_name)
        ds_args = load_config(f"datasets/{args.dataset_name}.json")

        # This is needed, else the logging file is not made (in PyCharm)
        # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if hasattr(args, 'seed'):
            seed = args.seed
        else:
            seed = 12

        # Unique name for this model and training method
        unique_name = generate_unique_name(args, ds_args)
        unique_name += f"_seed_{seed}"
        unique_name = "test13"

        switch_list = [1.0]
        if hasattr(args, 'switch_list'):
            switch_list = args.switch_list

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
        if args.dataset_name == "ImageNet2012":
            ds_train, ds_test = datasets['train'], datasets['validation']
        else:
            ds_train, ds_test = datasets['train'], datasets['test']

        # Build the input pipeline
        ds_train = ds_train.map(lambda x: {
            "image": preprocess_images_fn(args, ds_args, is_training=True)(x['image']),
            "label": x['label'],
        }).shuffle(args.batch_size * 50).batch(args.batch_size)

        ds_test = ds_test.map(lambda x: {
            "image": preprocess_images_fn(args, ds_args, is_training=False)(x['image']),
            "label": x['label'],
        }).shuffle(args.batch_size * 50).batch(1000)

        # Number of decay steps for training
        # decay_steps = args.num_epochs * int(ds_args.train_size / args.batch_size)
        # print(f"Decay steps = {decay_steps}")

        train_iterator = ds_train.make_initializable_iterator()
        next_train_element = train_iterator.get_next()

        test_iterator = ds_test.make_initializable_iterator()
        next_test_element = test_iterator.get_next()

        with tf.variable_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, ds_args.img_width, ds_args.img_height, ds_args.num_channels],
                               name="input_node")
            y = tf.placeholder(tf.float32, shape=[None, ds_args.num_classes])
            is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
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
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        )

        # Session, Interactive session (InteractiveSession)) catches errors better
        with tf.Session(config=config) as sess:
            sess.run(init_op)

            # Save the labels for this dataset
            with open(f"{conf.labels_dir}/{args.dataset_name}.txt", 'w') as f:
                for item in label_names:
                    f.write("%s\n" % item)

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
                        images = normalize_images(images, ds_args.mean, ds_args.std)

                        # One hot encode
                        labels = np.eye(ds_args.num_classes)[labels]

                        sw_idx, sw = random.choice(list(enumerate(switch_list)))
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

                            # Log to file
                            logging.info(f"TRAIN epoch: {epoch}/{args.num_epochs}, step {step}, train_loss: {avg_loss}, "
                                         f"train_acc: {avg_acc}, lr: {lr}")

                    except tf.errors.OutOfRangeError:
                        # print(f"Batch num = {num_batch}, Num images seen = {num_batch * args.batch_size}")
                        break

                # Auto detect problems and generate advice.
                # profiler.advise(options=opts)

                # Perform learning rate annealing
                lr = anneal_learning_rate(lr, epoch, step, args, sess, num_epochs=args.num_epochs)

                # ---------------- TESTING ---------------- #
                best_acc = 0
                if epoch % args.test_every == 0:

                    sess.run(test_iterator.initializer)
                    # Check results on all switches
                    for sw_idx, sw in enumerate(switch_list):
                        test_loss = []
                        test_acc = []
                        i = 0
                        while True:
                            try:
                                # Do profiling during test stage
                                batch = sess.run(next_test_element)
                                images, labels = batch['image'], batch['label']

                                images = images / 255.0
                                images = normalize_images(images, ds_args.mean, ds_args.std)

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

                        # Save the variables to disk.
                        last_ckpt = f"{checkpoint_dir}/epoch{epoch}.ckpt"
                        saver.save(sess, last_ckpt)

                        if avg_acc > best_acc:
                            # Found a new best validation accuracy model
                            last_ckpt = f"{checkpoint_dir}/best_epoch{epoch}_acc{avg_acc:.2f}.ckpt"
                            saver.save(sess, last_ckpt)
                            best_acc = avg_acc

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

            # ---------------- EXPORTING MODEL ---------------- #
            # region ExportModel
            print(f"Before map {sess.graph.get_tensor_by_name('input/is_training:0')}")
            # Export the graph
            save_graph_path = f"{conf.save_dir}/{unique_name}"
            if not os.path.exists(save_graph_path):
                # If this path does not exist, make it.
                os.makedirs(save_graph_path)

            tf.train.write_graph(sess.graph_def, save_graph_path, "graph_train.pb", as_text=False)

        # Freeze the graph
        freeze_graph.freeze_graph(input_graph=f"{save_graph_path}/graph_train.pb",
                                  input_saver="",
                                  input_binary=True,
                                  input_checkpoint=last_ckpt,
                                  output_node_names="network/output_node",
                                  restore_op_name="save/restore_all",
                                  filename_tensor_name="save/Const:0",
                                  output_graph=f"{save_graph_path}/frozen_model_train.pb",
                                  clear_devices=True,
                                  initializer_nodes="")

        # Starting a new session
        with tf.Session() as sess_eval:
            # Load frozen graph
            with gfile.FastGFile(f"{save_graph_path}/graph_train.pb", 'rb') as f:
                gd = tf.GraphDef()
                gd.ParseFromString(f.read())

            # Make constant False value (name does not need to match)
            is_training_const = tf.constant(False, dtype=tf.bool)
            # Load graph mapping placeholder to constant
            tf.import_graph_def(gd, input_map={"input/is_training:0": is_training_const})
            print(f"After map {sess_eval.graph.get_tensor_by_name('input/is_training:0')}")

            # Save graph again but with is_training fixed to false
            tf.train.write_graph(sess_eval.graph_def, save_graph_path, "graph_eval.pb", as_text=False)
            freeze_graph.freeze_graph(input_graph=f"{save_graph_path}/graph_eval.pb",
                                      input_saver="",
                                      input_binary=True,
                                      input_checkpoint=last_ckpt,
                                      output_node_names="network/output_node",
                                      restore_op_name="save/restore_all",
                                      filename_tensor_name="save/Const:0",
                                      output_graph=f"{save_graph_path}/frozen_model_eval.pb",
                                      clear_devices=True,
                                      initializer_nodes="")

            # Export model tflite
            # export_tflite_from_frozen_graph(f"{save_graph_path}/frozen_model_eval.pb",
            #                                input_nodes=["input/input_node"], output_nodes=["network/output_node"],
            #                                export_path=save_graph_path)
            # endregion
            logging.info("Finished")

