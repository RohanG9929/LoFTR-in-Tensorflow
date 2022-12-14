# LoFTR-in-Tensorflow

### [Project Page](https://rohang9929.github.io/loftr-in-tensorflow/) | [Presentation](https://youtu.be/Gs9Jz6svLko)

In an attempt to make the LoFTR [[1]](#1) algorithm more accessible, we have reimplmeneted it in Tensorflow.



## Comparison

Below is a comparison of the original LoFTR feature detector (top), and our reimplementation (bottom)

<img src="https://user-images.githubusercontent.com/103215628/206869093-ceb952d8-0804-4c7c-aa0c-5992d2cf422e.png" width="500" height="200"> 
<img src="https://user-images.githubusercontent.com/103215628/206869097-229f213d-c249-4c79-ac99-35083e17f1d9.png" width="500" height="200">



It is clear to see that our implementation is in need of more training. Due to time and computing resource constraints a full training could not be executed. Hence, the difference in results.

## Install

git clone link

cd LoFTR-in-Tensorflow

conda env create -f environment.yaml

## Usage

conda activate loftr_tf

### Demo Notebook ###

Run running.ipynb to see a visualisation of the LoFTR feature matcher running with our pretrained weights and some demo images.

## Training

Training was performed on 3 datasets. 

Megadepth [[2]](#2)

Scannet [[3]](#3)

NYU Depth V2 [[4]](#4)

See the [Training](./docs/Training.md) readme for details.

## Next Steps

Below are a few next steps to further improve this reimplementation.

1. Distrubute dataset using Tensorflows dataset builder to use multiple CPU cores to ush the data to multiple GPU's

2. Optimise Tensorflows GPU multi-worker strategy to work smoothly with our model



## References
<a id="1">[[1]](#1)</a> 
J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou (2021). 
LoFTR: Detector-free local feature matching with transformers

<a id="2">[[2]](#2)</a> 
Zhengqi Li and Noah Snavely (2018).
MegaDepth: Learning Single-View Depth Prediction from Internet Photos

<a id="3">[[3]](#3)</a> 
Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nießner, Matthias (2017). 
ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes

<a id="4">[[4]](#4)</a> 
Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus (2012). 
Indoor Segmentation and Support Inference from RGBD Images
