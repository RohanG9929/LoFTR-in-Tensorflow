# LoFTR-in-Tensorflow

In an attempt to make the LoFTR [[1]](#1) algorithm more accessible, we have reimplmeneted it in Tensorflow.

## Comparison

## Install

git clone link

cd loftr_tf

conda env create -f environment.yaml

## Usage

conda activate loftr_tf

python3 demo_loftr_tf.py --image1 path/image1.jpg --image2 path/image2.jpg

## Training

First, download the respective datasets from the links below:
megadepth (download the sfm AND the main file)
scannet (please request access via their github)
NYU Depth V2 (download _)

Download the indices from the original LoFTR paper.

Place each dataset in "file path to datasets folder"

place each indices folder in "file path to indices folder"

Download the weights from "our drive link"

python3 training_loftr_tf.py --dataset= "MegaDepth/Scannet/NYUDepth"

## Next Steps


## References
<a id="1">[[1]](#1)</a> 
J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou (2021). 
LoFTR: Detector-free local feature matching with transformers
