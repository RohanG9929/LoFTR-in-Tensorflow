import cv2 as cv
import tensorflow as tf
import h5py
print(tf.__version__)
print(tf.keras.__version__)
config = {'backbone_type': 'ResNetFPN', 
            'resolution': (8, 2), 
            'fine_window_size': 5, 
            'fine_concat_coarse_feat': True, 
            'resnetfpn': {'initial_dim': 128, 'block_dims': [128,196,256]}, 
            'coarse': {'d_model': 256, 'd_ffn': 256, 'nhead': 8, 'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'], 'attention': 'linear', 'temp_bug_fix': True}, 
            'match_coarse': {'thr': 0.2, 'border_rm': 2, 'match_type': 'dual_softmax', 'dsmax_temperature': 0.1, 'skh_iters': 3, 'skh_init_bin_score': 1.0, 'skh_prefilter': True, 'train_coarse_percent': 0.4, 'train_pad_num_gt_min': 200}, 
            'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self','cross'], 'attention': 'linear'}}


_config = {'loftr': {'backbone_type': 'ResNetFPN', 'resolution': (...), 'fine_window_size': 5, 'fine_concat_coarse_feat': True, 'resnetfpn': {...}, 'coarse': {...}, 'match_coarse': {...}, 'fine': {...}, 'loss': {...}}, 'dataset': {'trainval_data_source': 'ScanNet', 'train_data_root': 'data/scannet/train', 'train_pose_root': None, 'train_npz_root': 'data/scannet/index/s...data/train', 'train_list_path': 'data/scannet/index/s...et_all.txt', 'train_intrinsic_path': 'data/scannet/index/i...insics.npz', 'val_data_root': 'data/scannet/test', 'val_pose_root': None, 'val_npz_root': 'assets/scannet_test_1500', ...}, 'trainer': {'world_size': 4, 'canonical_bs': 64, 'canonical_lr': 0.006, 'scaling': 0.0625, 'find_lr': False, 'optimizer': 'adamw', 'true_lr': 0.000375, 'adam_decay': 0.0, 'adamw_decay': 0.1, ...}}


from LoFTR_TF import LoFTR

#Creating the matcher 
matcher = LoFTR(config)
import os

print(os.getcwd())


#loading in the images for the current batch
img0_pth = "./scene0738_00_frame-000885.jpg"
img1_pth = "./scene0738_00_frame-001065.jpg"
img0_raw = cv.imread(img0_pth, cv.IMREAD_GRAYSCALE)
img1_raw = cv.imread(img1_pth, cv.IMREAD_GRAYSCALE)
img0_raw = cv.resize(img0_raw, (640, 480))
img1_raw = cv.resize(img1_raw, (640, 480))

img0 = tf.convert_to_tensor(img0_raw)[None][None]/255
img1 = tf.convert_to_tensor(img1_raw)[None][None]/255

data = {'image0': img0, 'image1': img1}

#Calling the matcher on the current batch
updata = matcher(data)

print(updata)


