
from LoFTR_TF import LoFTR

config = {'backbone_type': 'ResNetFPN', 
            'resolution': (8, 2), 
            'fine_window_size': 5, 
            'fine_concat_coarse_feat': True, 
            'resnetfpn': {'initial_dim': 128, 'block_dims': [128,196,256]}, 
            'coarse': {'d_model': 256, 'd_ffn': 256, 'nhead': 8, 'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'], 'attention': 'linear', 'temp_bug_fix': True}, 
            'match_coarse': {'thr': 0.2, 'border_rm': 2, 'match_type': 'dual_softmax', 'dsmax_temperature': 0.1, 'skh_iters': 3, 'skh_init_bin_score': 1.0, 'skh_prefilter': True, 'train_coarse_percent': 0.4, 'train_pad_num_gt_min': 200}, 
            'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self','cross'], 'attention': 'linear'}}


_config = {'loftr': {'backbone_type': 'ResNetFPN', 'resolution': (...), 'fine_window_size': 5, 'fine_concat_coarse_feat': True, 'resnetfpn': {...}, 'coarse': {...}, 'match_coarse': {...}, 'fine': {...}, 'loss': {...}}, 'dataset': {'trainval_data_source': 'ScanNet', 'train_data_root': 'data/scannet/train', 'train_pose_root': None, 'train_npz_root': 'data/scannet/index/s...data/train', 'train_list_path': 'data/scannet/index/s...et_all.txt', 'train_intrinsic_path': 'data/scannet/index/i...insics.npz', 'val_data_root': 'data/scannet/test', 'val_pose_root': None, 'val_npz_root': 'assets/scannet_test_1500', ...}, 'trainer': {'world_size': 4, 'canonical_bs': 64, 'canonical_lr': 0.006, 'scaling': 0.0625, 'find_lr': False, 'optimizer': 'adamw', 'true_lr': 0.000375, 'adam_decay': 0.0, 'adamw_decay': 0.1, ...}}



#Creating the matcher 
matcher = LoFTR(config)


