"""
Main experimenting logic
"""
import os
import sys
# sys.path.insert(0, '~/intel/openvino_2019.1.085/python/python3.5')

import time
import cv2
import numpy as np
import logging as log

# pylint: disable=no-name-in-module
from openvino.inference_engine import IENetwork, IEPlugin

from tf_specific import load_graph, get_refs, parse_od_output
from common import draw_image, read_resize_image, show_results_interactively


def tf_main(path_to_model, path_to_original_image, path_to_result_image):
    """
    Entrypoint for inferencing with TensorFlow
    """
    log.info('COMMON: image preprocessing')
    width = 300
    resized_image = read_resize_image(path_to_original_image, width, width)
    reshaped_image = np.reshape(resized_image, (1, width, width, 3))

    log.info('TENSORFLOW SPECIFIC: Loading a model with TensorFLow')
    graph = load_graph(path_to_model)

    input_data = {
        'image_tensor': reshaped_image,
    }

    raw_results, delta = get_refs(graph, input_data)
    log.info('TENSORFLOW SPECIFIC: Plain inference finished')

    log.info('TENSORFLOW SPECIFIC: Post processing started')
    processed_results = parse_od_output(raw_results)
    log.info('TENSORFLOW SPECIFIC: Post processing finished')

    return processed_results['tf_detections'], delta


def ie_main(path_to_model_xml, path_to_model_bin, path_to_original_image,
            path_to_result_image, device='CPU', cpu_extensions=''):
    log.info('COMMON: image preprocessing')
    image = read_resize_image(path_to_original_image, 300, 300)

    log.info("Initializing plugin for {} device...".format(device))
    plugin = IEPlugin(device)

    if 'CPU' == device:
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

    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net

    labels_map = None

    image = image[..., ::-1]
    in_frame = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))

    inference_start = time.time()
    res = exec_net.infer(inputs={input_blob: in_frame})
    inference_end = time.time()

    log.info('INFERENCE ENGINE SPECIFIC: no post processing')

    return res[out_blob], inference_end - inference_start


if __name__ == '__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    IMAGE = './data/images/input/cat_on_snow.jpg'

    SSD_ASSETS = './data/object_detection/common/ssd_mobilenet_v2_coco/tf/'

    TF_MODEL = os.path.join(SSD_ASSETS, 'ssd_mobilenet_v2_coco.frozen.pb')
    TF_RESULT_IMAGE = './data/images/output/tensorflow_output.png'

    IE_MODEL_XML = os.path.join(SSD_ASSETS, 'ssd_mobilenet_v2_coco_ir.xml')
    IE_MODEL_BIN = os.path.join(SSD_ASSETS, 'ssd_mobilenet_v2_coco_ir.bin')
    IE_RESULT_IMAGE = './data/images/output/inference_engine_output.png'

    OPENVINO = '/home/demidovs/intel/openvino_2019.1.085/'
    OPENVINO_EXTENSIONS = 'deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
    IE_EXTENSIONS = os.path.join(OPENVINO, OPENVINO_EXTENSIONS)

    COMBO_RESULT_IMAGE = './data/images/output/combo_output.png'

    log.info('COMMON PART: Preparing the image')

    predictions, inf_time = tf_main(TF_MODEL, IMAGE, TF_RESULT_IMAGE)
    tf_fps = 1 / inf_time
    log.info('[TENSORFLOW] FPS: {}'.format(tf_fps))

    draw_image(IMAGE, predictions, TF_RESULT_IMAGE, color=(255, 0, 0))

    predictions, inf_time = ie_main(IE_MODEL_XML,
                                    IE_MODEL_BIN,
                                    IMAGE,
                                    IE_RESULT_IMAGE,
                                    'CPU',
                                    cpu_extensions=IE_EXTENSIONS)
    ie_fps = 1 / inf_time
    log.info('[INFERENCE ENGINE] FPS: {}'.format(ie_fps))

    draw_image(IMAGE, predictions, IE_RESULT_IMAGE, color=(0, 0, 255))

    draw_image(TF_RESULT_IMAGE, predictions, COMBO_RESULT_IMAGE, color=(0, 0, 255))

    show_results_interactively(tf_image=TF_RESULT_IMAGE,
                               ie_image=IE_RESULT_IMAGE,
                               combination_image=COMBO_RESULT_IMAGE,
                               ie_fps=ie_fps,
                               tf_fps=tf_fps)
