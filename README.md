# ObjectDetection_iUTS

To use the code, run `main.py` from your IDE or terminal. 
This file can run detection, model training, or open videos, depending on arguments that are passed to it.
Default values can be modified in `main.py`.  
```
(CompVision) soheil@Soheils-MacBook-Air codes % python main.py -h           
usage: main.py [-h] [-a {detect,train,open_video}] [-m MODEL] [-c CONFIG]

A program that runs computer vision tracking on videos in the data/input folder, with a focus on vehicle detection.

options:
  -h, --help            show this help message and exit
  -a {detect,train,open_video}, --action {detect,train,open_video}
                        The action to be performed by this script. default: detect
  -m MODEL, --model MODEL
                        The ID or path of the model to use. Default is resources/models/1.pt.
  -c CONFIG, --config CONFIG
                        Path to the config YAML file

```

## Models:
- model0: yoloV8? v11? v13?
- model1: ask Yiran about the soruce
- model2: trained using 100 fisheye annotations
- model3: 

```
conda create -n objectDetection python=3.10
conda activate objectDetection
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics

```