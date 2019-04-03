# openvino-tf-experiment

1. Setup Python environment

2. Install dependencies

    2.1. Install OpenVINO

    2.2. Setup environment
    
    ```
    cd ~/intel/openvino/bin
    source setupvars.sh
    ```

    2.3. Install Python API dependencies:

    ```
    cd ~/intel/openvino/python
    pip3 install -r requirements.txt
    export PYTHONPATH=~/intel/openvino/python/python3.6/:$PYTHONPATH
    ```

    2.4. Install local dependencies:
    
    ```
    pip3 install -U matplotlib
    sudo apt-get install python3-tk
    ```

3. Download the TensorFlow Mobilenet model:

```
cd ~/intel/openvino/deployment_tools/tools/model_downloader/
pip3 install requests
python3 downloader.py --name ssd_mobilenet_v2_coco -o ~/Projects/openvino-tf-experiment/data
```

Also download the `.config` file, describing the structure of the SSD head of the model:
```
wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config -O ~/Projects/openvino-tf-experiment/data/object_detection/common/ssd_mobilenet_v2_coco/tf/ssd_mobilenet_v2_coco.config
```

It is now placed in the `~/Projects/openvino-tf-experiment/data/object_detection/common/ssd_mobilenet_v2_coco/tf`.

4. If you want, inspect it with [Netron](https://lutzroeder.github.io/netron/) or Tensorboard.

5. Infer the model on TensorFlow.

6. Read more about the way a model is prepared for inference and what is Inference Engine IR (Intermediate Representation) format. 
   Explore [the format](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_IRLayersCatalogSpec.html)
   and [TensorFlow supported layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)

7. Convert a model to IR with Model Optimizer tool
   
   The exact Model Optimizer command is taken from `~/intel/openvino/deployment_tools/tools/model_downloader/list_topologies.yml`.
   Find by name of the model: `ssd_mobilenet_v2_coco`

```
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

8. Infer the model on Inference Engine.
