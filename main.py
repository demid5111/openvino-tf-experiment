"""
Main experimenting logic
"""
import os
import sys
import platform
# sys.path.insert(0, '~/intel/openvino_2019.3.334/python/python3.5')

import time
import cv2
import numpy as np
import logging as log

# pylint: disable=no-name-in-module
from openvino.inference_engine import IENetwork, IEPlugin

from tf_specific import load_graph, get_refs, parse_od_output
from common import draw_image, read_resize_image, show_results_interactively


def tf_main(path_to_model, path_to_original_image, path_to_result_image, batch = 1):
    """
    Entrypoint for inferencing with TensorFlow
    """
    log.info('COMMON: image preprocessing')
    width = 300
    resized_image = read_resize_image(path_to_original_image, width, width)
    reshaped_image = np.reshape(resized_image, (width, width, 3))
    batched_image = np.array([reshaped_image for _ in range(batch)])
    log.info('Current shape: {}'.format(batched_image.shape))

    log.info('TENSORFLOW SPECIFIC: Loading a model with TensorFLow')
    graph = load_graph(path_to_model)

    input_data = {
        'image_tensor': batched_image,
    }

    raw_results, delta = get_refs(graph, input_data)
    log.info('TENSORFLOW SPECIFIC: Plain inference finished')

    log.info('TENSORFLOW SPECIFIC: Post processing started')
    processed_results = parse_od_output(raw_results)
    log.info('TENSORFLOW SPECIFIC: Post processing finished')

    return processed_results['tf_detections'], delta


def ie_main(path_to_model_xml, path_to_model_bin, path_to_original_image,
            path_to_result_image, device='CPU', cpu_extensions='', batch=1):
    log.info('COMMON: image preprocessing')
    image = read_resize_image(path_to_original_image, 300, 300)

    log.info("Initializing plugin for {} device...".format(device))
    plugin = IEPlugin(device)

    if 'CPU' == device and cpu_extensions:
        plugin.add_cpu_extension(cpu_extensions)

    log.info("Reading IR...")
    net = IENetwork(model=path_to_model_xml, weights=path_to_model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            sys.exit(1)
    assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
    assert len(net.outputs) == 1, "Demo supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.inputs[input_blob].shape
    net.reshape({input_blob: (batch, c, h, w)})
    n, c, h, w = net.inputs[input_blob].shape

    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    del net

    labels_map = None
    
    # Read and pre-process input image
    image = image[..., ::-1]
    in_frame = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    batched_frame = np.array([in_frame for _ in range(batch)])
    log.info('Current shape: {}'.format(batched_frame.shape))

    inference_start = time.time()
    res = exec_net.infer(inputs={input_blob: batched_frame})
    inference_end = time.time()

    log.info('INFERENCE ENGINE SPECIFIC: no post processing')

    return res[out_blob], inference_end - inference_start


if __name__ == '__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    NUM_RUNS = 1
    BATCH = 1

    IMAGE = './data/images/input/cat_on_snow.jpg'

    SSD_ASSETS = './data/public/ssd_mobilenet_v2_coco'

    TF_MODEL = os.path.join(SSD_ASSETS, 'ssd_mobilenet_v2_coco_2018_03_29', 'frozen_inference_graph.pb')
    TF_RESULT_IMAGE = './data/images/output/tensorflow_output.png'

    IE_MODEL_XML = os.path.join(SSD_ASSETS, 'FP32', 'ssd_mobilenet_v2_coco.xml')
    IE_MODEL_BIN = os.path.join(SSD_ASSETS, 'FP32', 'ssd_mobilenet_v2_coco.bin')
    IE_RESULT_IMAGE = './data/images/output/inference_engine_output.png'

    OPENVINO = '/home/dev/intel/openvino'
    OPENVINO_EXTENSIONS_DIR = 'deployment_tools/inference_engine/samples/build/intel64/Release/lib/'
    
    if platform.system() == 'Darwin':
        ext = '.dylib'
    elif platform.system() == 'Linux':
        ext = '.so'
    else:
        print('You are running this demo on Windows OS. However, this is demo for Linux/macOS.')
        sys.exit(0)

    OPENVINO_EXTENSIONS = 'libcpu_extension{}'.format(ext)
    IE_EXTENSIONS = os.path.join(OPENVINO, OPENVINO_EXTENSIONS)

    COMBO_RESULT_IMAGE = './data/images/output/combo_output.png'

    log.info('COMMON PART: Preparing the image')

    tf_fps_collected = []
    for i in range(NUM_RUNS):
        predictions, inf_time = tf_main(TF_MODEL, IMAGE, TF_RESULT_IMAGE, batch=BATCH)
        tf_fps = 1 / inf_time
        tf_fps_collected.append(tf_fps)
    
    tf_avg_fps = (sum(tf_fps_collected) * BATCH) / (NUM_RUNS)
    log.info('[TENSORFLOW] FPS: {}'.format(tf_avg_fps))
    
    draw_image(IMAGE, predictions, TF_RESULT_IMAGE, color=(255, 0, 0))

    ie_fps_collected = []
    for i in range(NUM_RUNS):
        predictions, inf_time = ie_main(IE_MODEL_XML,
                                        IE_MODEL_BIN,
                                        IMAGE,
                                        IE_RESULT_IMAGE,
                                        'CPU',
                                        cpu_extensions=IE_EXTENSIONS,
                                        batch=BATCH)
        ie_fps = 1 / inf_time
        ie_fps_collected.append(ie_fps)
    
    ie_avg_fps = (sum(ie_fps_collected) * BATCH) / (NUM_RUNS)
    log.info('[INFERENCE ENGINE] FPS: {}'.format(ie_avg_fps))

    draw_image(IMAGE, predictions, IE_RESULT_IMAGE, color=(0, 0, 255))

    draw_image(TF_RESULT_IMAGE, predictions, COMBO_RESULT_IMAGE, color=(0, 0, 255))

    show_results_interactively(tf_image=TF_RESULT_IMAGE,
                               ie_image=IE_RESULT_IMAGE,
                               combination_image=COMBO_RESULT_IMAGE,
                               ie_fps=ie_avg_fps,
                               tf_fps=tf_avg_fps)
