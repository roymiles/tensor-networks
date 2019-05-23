
from Networks.convert import standard_to_tucker
import tensorflow as tf
import tensorflow_datasets as tfds


def convert_example(model, x, sess, conv_ranks, fc_ranks):
    """
    An example of how to convert standard network to tucker network.
    It was getting a bit messy putting this in the train scripts

    :param model: Model type StandardNetwork
    :param x: Input placeholder
    :param sess: Current tensorflow session
    :param conv_ranks: Convolutional layer ranks
    :param fc_ranks: Fully connected layer ranks
    :return:
    """

    # Find a reasonable approximation to the weights given these ranks (minimise reconstruction error)
    converted_model, sess, merged = standard_to_tucker(model, sess, conv_ranks, fc_ranks)

    convert_writer = tf.summary.FileWriter('/home/roy/Desktop/Tensorboard/convert', sess.graph)
    print("Run: \"tensorboard --logdir=/home/roy/Desktop/Tensorboard/convert\"")

    print("Number of parameters (converted) = {}".format(converted_model.num_parameters()))

    # Single forward pass (of the converted model)
    logits = converted_model(input=x)

    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)
    avg_loss = tf.reduce_mean(loss_op)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    end_points = {'logits': logits, 'accuracy': accuracy}

    return end_points

    """
    # Now test the accuracy on the converted model
    for batch in tfds.as_numpy(ds_test):
        images, labels = batch['image'], batch['label']

        # Normalise in range [0, 1)
        images = images / 255.0

        # One hot encode
        labels = np.eye(10)[labels]

        feed_dict = {
            x: images,
            y: labels
        }

        acc, summary = sess.run([accuracy, merged], feed_dict)
        print("Accuracy (converted) = {}".format(acc))
        convert_writer.add_summary(summary)
    """