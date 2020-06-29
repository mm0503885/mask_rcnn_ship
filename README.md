# Instance Segmentation for ships in satellite imagery

## Introduction
This is an Instance Segmentation task for [ships in satellite imagery](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) based on  [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

## File structure 
* (mrcnn) Source code of Mask R-CNN built on FPN and ResNet101.
* (samples\ship\inspect_ship_model.ipynb) Jupyter notebooks to visualize the detection and compute mAP
* (samples\ship\ship.py) training code on ships in satellite imagery

## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

## How to execute
Train a new model starting from pre-trained COCO weights
```
python3 samples\ship\ship.py train --dataset=/path/to/ship/dataset --weights=coco
```
Resume training a model that you had trained earlier
```
python3 samples\ship\ship.py train --dataset=/path/to/ship/dataset --weights=last
```

## Run Jupyter notebooks
Open the  `inspect_ship_model.ipynb` Jupter notebooks. You can use these notebooks to run through the detection pipelie step by step and compute mAP.
