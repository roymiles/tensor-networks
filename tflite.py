""" Provides useful utility functions for converting models to tflite format """

import tensorflow as tf
import numpy as np


def export_tflite(sess, input_nodes, output_nodes, optimizations, name):
    """ Convert a sess graph to .tflite file
        input_nodes and output_nodes are arrays of Tensors for the input
        and output variables

        :param optimizations: Array of optimizations e.g. [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        :param name: Name of output tflite file """

    sess.run(tf.global_variables_initializer())
    converter = tf.lite.TFLiteConverter.from_session(sess, input_nodes, output_nodes)
    converter.optimizations = optimizations
    tflite_model = converter.convert()
    open("{}.tflite".format(name), "wb").write(tflite_model)


def export_tflite(saved_model_dir, optimizations, name):
    """ Convert saved model to tflite

        :param optimizations: Array of optimizations e.g. [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        :param name: Name of output tflite file
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = optimizations
    tflite_model = converter.convert()
    open("{}.tflite".format(name), "wb").write(tflite_model)


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
