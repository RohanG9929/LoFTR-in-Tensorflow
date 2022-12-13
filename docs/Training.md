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

1. The NYU Depth V2 is provided in a) labeled format b) raw format. In the project, we downloaded several scenes from the raw format as it suits better our needs for the final data structure. the download link for the dataset is provided below.
https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

2. After downloading the scenes, we select the corresponding ppm and pgm files from the index.txt file.

3. The depth files (pgm) files are not provided in the frame of the RGB camera, and need to undergo transformation to the same frame. The toolbox provided with the dataset contains MATLAB scripts to transform the depth maps to align with the RGB frames. The download link for NYU MATLAB toolbox is provided below

http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip

4. After transforming the depth maps, we needed to select corresponding pairs of images that respect the condition of the covisibility being between 0.4 and 0.7. Python code was developed by using SIFT features detector to select such pairs of images and the corresponding depth maps.

5. The output from the python file generates a list of dictionaries containing batch of 4 pairs of images, depth, transformation and corresponding intrinsic matrices. 









