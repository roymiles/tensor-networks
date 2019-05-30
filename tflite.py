""" Provides useful utility functions for converting models to tflite format """

import tensorflow as tf
import numpy as np
import config as conf


def export_tflite_from_session(sess, input_nodes, output_nodes, name, optimizations=None):
    """ Convert a sess graph to .tflite file
        input_nodes and output_nodes are arrays of Tensors for the input
        and output variables

        :param optimizations: Array of optimizations e.g. [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        :param name: Name of output tflite file """

    sess.run(tf.global_variables_initializer())
    converter = tf.lite.TFLiteConverter.from_session(sess, input_nodes, output_nodes)
    converter.allow_custom_ops = True

    if optimizations:
        converter.optimizations = optimizations

    tflite_model = converter.convert()

    export_path = "{}/{}.pbtxt".format(conf.tflite_dir, name)
    open(export_path, "wb").write(tflite_model)
    print("Successfully exported to: {}".format(export_path))


def export_tflite_from_saved_model(saved_model_dir, name, optimizations=None):
    """ Convert saved model to tflite

        :param saved_model_dir: Path to the saved model
        :param optimizations: Array of optimizations e.g. [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        :param name: Name of output tflite file
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.allow_custom_ops = True

    if optimizations:
        converter.optimizations = optimizations

    tflite_model = converter.convert()
    export_path = "{}/{}.pbtxt".format(conf.tflite_dir, name)
    open(export_path, "wb").write(tflite_model)
    print("Successfully exported to: {}".format(export_path))


def invoke_tflite(model_path, input_data):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    assert input_shape == input_data.shape, "Invalid shape for the input node"
    # TODO: Check data type too

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
