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
   git clone https://github.com/demid5111/openvino-tf-experiment ~/Projects/openvino-tf-experiment
   ```

2. Setup Python environment. Install Python 3.5+.

3. Install dependencies

    2.1. Install OpenVINO 2019 R3. Take it from 
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
    sudo -E apt-get install python3-tk
    ```

    2.5 Build extensions:
       ```bash
       cd ~/intel/openvino/deployment_tools/inference_engine/samples
       mkdir build
       cd build
       cmake ..
       make ie_cpu_extension -j8
       ```

## Download the TensorFlow model <a name="download"></a>

1. Download the TensorFlow Mobilenet model:

   ```bash
   cd ~/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
   pip3 install requests
   python3 downloader.py --name ssd_mobilenet_v2_coco -o ~/Projects/openvino-tf-experiment/data
   ```

2. If you want, inspect it with [Netron](https://lutzroeder.github.io/netron/) or Tensorboard.

## Infer the model on TensorFlow <a name="tf-infer"></a>

* Lines 20-43 correspond to the `tf_main` function that performs inference.
* Lines 115-119 run the inference and create the output image.

## Create Intermediate Representation of the model (IR) for Inference Engine <a name="convert"></a>

1. Read more about the way a model is prepared for inference and what is Inference Engine IR (Intermediate Representation) format.
   Explore [the format](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_IRLayersCatalogSpec.html)
   and [TensorFlow supported layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)

2. Convert a model to IR. In general, you can do it only with Model Optimizer tool, however for Open Model Zoo models you can convert it
   with a wrapper over Model Optimizer that considerably simplifies the flow.
   
   1. Create IR via the higher level wrapper over Model Optimizer.
      ```bash
      cd ~/intel/openvino/deployment_tools/model_optimizer/
      pip3 install -r requirements.txt
      cd ~/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
      python3 converter.py -d ~/Projects/openvino-tf-experiment/data `# where to take original model from`\
                           --name ssd_mobilenet_v2_coco `# name of the original model`\
                           --precisions FP32 `# precision of the resulting model`
      ```
   2. Alternative option. Any model can be converted with Model Optimizer. However, it requires deep understanding of the model: 
      its inputs, shapes, normalization values etc. The same `ssd_mobilenet_v2_coco` can be converted with the following command:

      ```bash
      cd ~/intel/openvino/deployment_tools/model_optimizer/
      pip3 install -r requirements.txt
      python3 mo.py --framework tf `# we convert a TensorFlow model`\
                    --data_type FP32 `# it is trained in floating point 32-bit`\
                    --reverse_input_channels `# exchange channels from BGR to RGB`\
                    --input_shape [1,300,300,3] `# original model has dynamic shapes, specify ones that we need`\
                    --input image_tensor `# the name of the input layer`\
                    --tensorflow_use_custom_operations_config ./extensions/front/tf/ssd_v2_support.json `# Model Optimizer extensions for the model`\
                    --tensorflow_object_detection_api_pipeline_config ~/Projects/openvino-tf-experiment/data/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config `# TensorFlow Object Detection API config (standard and delivered with the model)`\
                    --output detection_classes,detection_scores,detection_boxes,num_detections `# output layers`\
                    --input_model /home/dev/Projects/openvino-tf-experiment/data/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb `# path to the model`\
                    --model_name ssd_mobilenet_v2_coco_ir `# name of the output model`\
                    --output_dir ~/Projects/openvino-tf-experiment/data/public/ssd_mobilenet_v2_coco/FP32 `# where to store resulting IR` 
      ```

## Infer the model on Inference Engine <a name="ie-infer"></a>

* Lines 46-90 correspond to the `ie_main` function that performs inference.
* Lines 121-130 run the inference and create the output image.
