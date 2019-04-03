# openvino-tf-experiment

## Contents

1. <a href="#prerequisites">Install prerequisites</a>
2. <a href="#download">Download the TensorFlow model</a>
3. <a href="#tf-infer">Infer the model on TensorFlow</a>
4. <a href="#convert">Create Intermediate Representation of the model (IR) for Inference Engine</a>
5. <a href="#ie-infer">Infer the model on Inference Engine</a>

## Install prerequisites <a name="prerequisites"></a>

> **NOTE**: This is the Ubuntu 16.04 tutorial. Should not be a problem for Ubuntu 18. For Windows there
> can be minor changes required due to specificity of using Python and related packages

1. Clone the repo

   ```bash
   git clone https://github.com/demid5111/openvino-tf-experiment
   ```

2. Setup Python environment. Install Python 3.5+.

3. Install dependencies

    2.1. Install OpenVINO 2019 R1. Take it from 
         [here](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux).
         All commands below assume that it was installed for a local user. Otherwise, slightly change paths

    2.2. Setup environment

    ```bash
    cd ~/intel/openvino/bin
    source setupvars.sh
    ```

    2.3. Install Python API dependencies. Note that `python3.6` directory was used. 
         If you have another Python version, apply accordingly:

    ```bash
    cd ~/intel/openvino/python
    pip3 install -r requirements.txt
    export PYTHONPATH=~/intel/openvino/python/python3.6/:$PYTHONPATH
    ```

    2.4. Install local dependencies:

    ```bash
    pip3 install -U -r requirements.txt --no-cache-dir
    sudo apt-get install python3-tk
    ```

## Download the TensorFlow model <a name="download"></a>

1. Download the TensorFlow Mobilenet model:

   ```bash
   cd ~/intel/openvino/deployment_tools/tools/model_downloader/
   pip3 install requests
   python3 downloader.py --name ssd_mobilenet_v2_coco -o ~/Projects/openvino-tf-experiment/data
   ```

2. Download the `.config` file, describing the structure of the SSD head of the model:

   ```bash
   wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config -O ~/Projects/openvino-tf-experiment/data/object_detection/common/ssd_mobilenet_v2_coco/tf/ssd_mobilenet_v2_coco.config
   ```

   It is now placed in the `~/Projects/openvino-tf-experiment/data/object_detection/common/ssd_mobilenet_v2_coco/tf`.

3. If you want, inspect it with [Netron](https://lutzroeder.github.io/netron/) or Tensorboard.

## Infer the model on TensorFlow <a name="tf-infer"></a>

* Lines 20-43 correspond to the `tf_main` function that performs inference.
* Lines 115-119 run the inference and create the output image.

## Create Intermediate Representation of the model (IR) for Inference Engine <a name="convert"></a>

1. Read more about the way a model is prepared for inference and what is Inference Engine IR (Intermediate Representation) format.
   Explore [the format](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_IRLayersCatalogSpec.html)
   and [TensorFlow supported layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)

2. Convert a model to IR with Model Optimizer tool.
   The exact Model Optimizer command is taken from `~/intel/openvino/deployment_tools/tools/model_downloader/list_topologies.yml`.
   Find by name of the model: `ssd_mobilenet_v2_coco`

   ```bash
   cd ~/intel/openvino/deployment_tools/model_optimizer/
   pip3 install -r requirements.txt
   python mo.py --framework tf `# we convert a TensorFlow model`\
                --data_type FP32 `# it is trained in floating point 32-bit`\
                --reverse_input_channels `# exchange channels from BGR to RGB`\
                --input_shape [1,300,300,3] `# original model has dynamic shapes, specify ones that we need`\
                --input image_tensor `# the name of the input layer`\
                --tensorflow_use_custom_operations_config ./extensions/front/tf/ssd_v2_support.json `# Model Optimizer extensions for the model`\
                --tensorflow_object_detection_api_pipeline_config ~/Projects/openvino-tf-experiment/data/object_detection/common/ssd_mobilenet_v2_coco/tf/ssd_mobilenet_v2_coco.config `# TensorFlow Object Detection API config (standard and delivered with the model)`\
                --output detection_classes,detection_scores,detection_boxes,num_detections `# output layers`\
                --input_model ~/Projects/openvino-tf-experiment/data/object_detection/common/ssd_mobilenet_v2_coco/tf/ssd_mobilenet_v2_coco.frozen.pb `# path to the model`\
                --model_name ssd_mobilenet_v2_coco_ir `# name of the output model`\
                --output_dir ~/Projects/openvino-tf-experiment/data/object_detection/common/ssd_mobilenet_v2_coco/tf/ `# where to store all the models`
   ```

## Infer the model on Inference Engine <a name="ie-infer"></a>

* Lines 46-90 correspond to the `ie_main` function that performs inference.
* Lines 121-130 run the inference and create the output image.
