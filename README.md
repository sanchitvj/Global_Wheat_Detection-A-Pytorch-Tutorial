# Object Detection Using Faster RCNN in Pytorch
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

## Introduction
This is a tutorial for beginners on how to train a [Faster RCNN](https://arxiv.org/abs/1506.01497) network for object detection in pytorch.
Dataset used is Global Wheat Detection available on [kaggle](https://www.kaggle.com/c/global-wheat-detection). The code here is commented with
explanations and customised visualizing functions for validation image output epoch-wise.

Note: Incase notebook is not loading on GitHub, you can check notebook with validation output upto 10 epochs [here](https://nbviewer.jupyter.org/github/sanchitvj/Global_Wheat_Detection-A-Pytorch-Tutorial/blob/master/notebook/global-wheat-detection.ipynb).
Code in python file has some issues so check notebook for smoothly working code.

**Please STAR ⭐ the repository.**

## Requirements
- Python(3.6+)
- Pytorch(1.7)
- GPU: Nvidia Tesla P100(provided by Kaggle)

## Try Yourself
Download the saved model from [here](https://drive.google.com/file/d/1cQX-RtX2uRqcub0iJbGlzLzRPjYjvwcE/view?usp=sharing).

## Validation Output Sample
![test1](https://github.com/sanchitvj/Global_Wheat_Detection-A-Pytorch-Tutorial/blob/master/validation_output/__results___23_7.png)
![test2](https://github.com/sanchitvj/Global_Wheat_Detection-A-Pytorch-Tutorial/blob/master/validation_output/__results___23_46.png)

## Test Output Sample
![test3](https://github.com/sanchitvj/Global_Wheat_Detection-A-Pytorch-Tutorial/blob/master/test_output/__results___31_5.png)
![test4](https://github.com/sanchitvj/Global_Wheat_Detection-A-Pytorch-Tutorial/blob/master/test_output/__results___31_1.png)
