# Training



First, download the respective datasets from the links below:

[Megadepth](https://www.cs.cornell.edu/projects/megadepth/ "Megadepth")

megadepth  (download the sfm AND the main file)
scannet (please request access via their github)
NYU Depth V2 (download _)

Download the indices from the original LoFTR paper.

Place each dataset in "file path to datasets folder"

place each indices folder in "file path to indices folder"

Download the weights from "our drive link"

python3 training_loftr_tf.py --dataset= "MegaDepth/Scannet/NYUDepth"
