"""
TensorFlow specific processing logic
"""
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.framework import graph_io

import numpy as np
import logging as log
import time


def load_graph(path_to_model):
    """
    Creates in memory graph in TensorFlow
    """
    tf.reset_default_graph()
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(path_to_model, "rb") as model_file:
        graph_def.ParseFromString(model_file.read())

    nodes_to_clear_device = graph_def.node if isinstance(
        graph_def, tf.GraphDef) else graph_def.graph_def.node
    for node in nodes_to_clear_device:
        node.device = ""

    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    log.info("tf graph was created")
    return graph


def children(op_name: str, graph: tf.Graph):
    """Get operation node children."""
    op = graph.get_operation_by_name(op_name)
    return set(op for out in op.outputs for op in out.consumers())


def summarize_graph(graph_def):
    unlikely_output_types = [
        'Const', 'Assign',
        'NoOp', 'Placeholder',
        'Assert', 'switch_t', 'switch_f'
    ]
    placeholders = dict()
    outputs = list()
    graph = tf.Graph()
    with graph.as_default():  # pylint: disable=not-context-manager
        tf.import_graph_def(graph_def, name='')
    for node in graph.as_graph_def().node:  # pylint: disable=no-member
        if node.op == 'Placeholder':
            node_dict = dict()
            node_dict['type'] = tf.DType(node.attr['dtype'].type).name
            new_shape = tf.TensorShape(node.attr['shape'].shape)
            node_dict['shape'] = str(new_shape).replace(' ', '').replace('?', '-1')
            placeholders[node.name] = node_dict
        if len(children(node.name, graph)) == 0:
            if node.op not in unlikely_output_types and \
                node.name.split('/')[-1] not in unlikely_output_types:
                outputs.append(node.name)
    result = dict()
    result['inputs'] = placeholders
    result['outputs'] = outputs
    return result


def get_refs(graph, input_data):
    """Return TensorFlow model reference results."""
    log.info("Running inference with tensorflow ...")
    feed_dict = {}
    summary_info = summarize_graph(graph.as_graph_def())
    input_layers, output_layers = list(summary_info['inputs'].keys()), summary_info['outputs']

    data_keys = [key for key in input_data.keys()]
    if sorted(input_layers) != sorted(data_keys):
        raise ValueError('input data keys: {0} do not match input '
                         'layers of network: {1}'.format(data_keys, input_layers))

    for input_layer_name in input_layers:
        tensor = graph.get_tensor_by_name(input_layer_name + ':0')
        feed_dict[tensor] = input_data[input_layer_name]
    output_tensors = []
    for name in output_layers:
        tensor = graph.get_tensor_by_name(name + ':0')
        output_tensors.append(tensor)

    log.info("Running tf.Session")
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # with tf.device("/cpu:0"):
    with graph.as_default():
        with tf.Session(graph=graph) as session:
            inference_start = time.time()
            outputs = session.run(output_tensors, feed_dict=feed_dict)
            inference_end = time.time()
    res = dict(zip(output_layers, outputs))
    log.info("TensorFlow reference collected successfully\n")
    return res, inference_end - inference_start


def parse_od_output(data: dict):
    predictions = []
    num_batches = len(data['detection_boxes'])
    target_layers = ['num_detections', 'detection_classes',
                     'detection_scores', 'detection_boxes']

    for b in range(num_batches):
        predictions.append([])
        num_detections = int(data['num_detections'][b])
        detection_classes = data['detection_classes'][b]
        detection_scores = data['detection_scores'][b]
        detection_boxes = data['detection_boxes'][b]
        for i in range(num_detections):
            obj = [
                b, detection_classes[i], detection_scores[i],
                detection_boxes[i][1], detection_boxes[i][0],
                detection_boxes[i][3], detection_boxes[i][2]
            ]
            predictions[b].append(obj)
    predictions = np.asarray(predictions)
    new_shape = (1, 1, predictions.shape[0] * predictions.shape[1], predictions.shape[2])
    predictions = np.reshape(predictions, newshape=new_shape)
    parsed_data = {'tf_detections': predictions}
    for layer, blob in data.items():
        if layer not in target_layers:
            parsed_data.update({layer: blob})
    return parsed_data
