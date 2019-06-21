""" Provides useful utility functions for converting models to tflite format """

import tensorflow as tf
import numpy as np
import config as conf
import os


def _export(converter, unique_name):
    """

    :param converter:
    :param unique_name: Unique name, includes dataset, architecture etc
    :return:
    """
    tflite_model = converter.convert()

    export_path = f"{conf.tfds_dir}/{unique_name}"
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    export_path = f"{export_path}/model.tflite"
    open(export_path, "wb").write(tflite_model)
    print("Successfully exported to: {}".format(export_path))


def freeze_graph(sess, output_node_names):
    """

    :param sess: Current session
    :param output_node_names: List of output node names
    :return: Frozen graph
    """
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        tf.get_default_graph().as_graph_def(),  # input_graph_def is useful for retrieving the nodes 
        output_node_names
    )

    return output_graph_def


def export_tflite_from_session(sess, input_nodes, output_nodes, unique_name, optimizations=None):
    """ Convert a sess graph to .tflite file
        input_nodes and output_nodes are arrays of Tensors for the input
        and output variables

        :param input_nodes: List of input tensors (aka placeholders)
        :param optimizations: Array of optimizations e.g. [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] """

    converter = tf.lite.TFLiteConverter.from_session(sess, input_nodes, output_nodes)
    converter.allow_custom_ops = True

    if optimizations:
        converter.optimizations = optimizations

    _export(converter, unique_name)


def export_tflite_from_frozen_graph(frozen_graph_def, input_nodes, output_nodes, unique_name, optimizations=None):
    converter = tf.lite.TFLiteConverter.from_frozen_graph(frozen_graph_def, input_nodes, output_nodes)
    converter.allow_custom_ops = True

    if optimizations:
        converter.optimizations = optimizations

    _export(converter, unique_name)


def export_tflite_from_saved_model(saved_model_dir, unique_name, optimizations=None):
    """ Convert saved model to tflite

        :param saved_model_dir: Path to the saved model
        :param optimizations: Array of optimizations e.g. [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.allow_custom_ops = True

    if optimizations:
        converter.optimizations = optimizations

    _export(converter, unique_name)


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
