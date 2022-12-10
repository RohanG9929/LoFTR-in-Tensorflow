# LoFTR-in-Tensorflow

In an attempt to make the LoFTR [[1]](#1) algorithm more accessible, we have reimplmeneted it in Tensorflow.

## Comparison

## Install

git clone link

cd LoFTR-in-Tensorflow

conda env create -f environment.yaml

## Usage

conda activate loftr_tf

### Demo Notebook ###

Run running.ipynb to see a visualisation of the LoFTR feature matcher running with some demo images.

## Training

First, download the respective datasets from the links below:

[Named Link](http://www.google.fr/ "Named link title")

[Megadepth](https://www.cs.cornell.edu/projects/megadepth/ "Megadepth")

megadepth  (download the sfm AND the main file)
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
