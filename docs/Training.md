# Training

Below are the steps to run 


## Megadepth



1. Download the datasets

  From the link below (download the MegaDepth v1 Dataset & MegaDepth v1 SfM models):
  [Megadepth](https://www.cs.cornell.edu/projects/megadepth/ "Megadepth") 

  Unzip the tar files and rename the MegaDepth v1 SfM folder to "Undistorted SfM"

2. Move datasets to correct folder

  Move both the v1 and SfM datasets into 
  "/LoFTR-in-Tensorflow/src/training/datasets/megadepth"

3. Download Indicies

  Download the indices from the original LoFTR paper. 

  [Megadepth Indices](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing "Megadepth Indices")

  After opening the link, the indices are located in "LoFTR/train-data/megadepth_indices.tar"

4. Move indices to correct folder

  Untar the indices folder and also move it to 
  "/LoFTR-in-Tensorflow/src/training/datasets/megadepth"


5. Run Training File

Currently the training file is a CPU only training file. We are working on a GPU training file that will utilise multiple GPU's

The weights for this training will be saved into "./weights/megadepth/cp_megadepth.ckpt"


## Scannet

scannet (please request access via their github)



## NYU Depth V2



