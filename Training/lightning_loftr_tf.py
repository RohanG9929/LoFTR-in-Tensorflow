# 1- define optimizer
# 2-one step through optimizer
# 3- train val inference
# 4-  training_step
# 5- training epoch end
# 6- compute metrics

import os
import sys
print(os.getcwd())
import tensorflow as tf
from LoFTR_TF import LoFTR
os.chdir("LoFTR-in-Tensorflow")
from supervisionTF import compute_supervision_coarse, compute_supervision_fine
from loftr_lossTF import LoFTRLoss
from loadData import read_data


config = {'LOFTR': 

{'BACKBONE_TYPE': 'ResNetFPN', 'RESOLUTION': (8, 2), 'FINE_WINDOW_SIZE': 5, 'FINE_CONCAT_COARSE_FEAT': True, 
    'RESNETFPN': {'INITIAL_DIM': 128, 'BLOCK_DIMS': [128, 196, 256]}, 
'COARSE': {'D_MODEL': 256, 'D_FFN': 256, 'NHEAD': 8, 'LAYER_NAMES': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'], 'ATTENTION': 'linear', 'TEMP_BUG_FIX': True}, 
'MATCH_COARSE': {'THR': 0.2, 'BORDER_RM': 2, 'MATCH_TYPE': 'dual_softmax', 'DSMAX_TEMPERATURE': 0.1, 'SKH_ITERS': 3, 'SKH_INIT_BIN_SCORE': 1.0, 'SKH_PREFILTER': False, 'TRAIN_COARSE_PERCENT': 0.2, 'TRAIN_PAD_NUM_GT_MIN': 200, 'SPARSE_SPVS': True},
'FINE': {'D_MODEL': 128, 'D_FFN': 128, 'NHEAD': 8, 'LAYER_NAMES': ['self', 'cross'], 'ATTENTION': 'linear'}, 
'LOSS': {'COARSE_TYPE': 'focal', 'COARSE_WEIGHT': 1.0, 'FOCAL_ALPHA': 0.25, 'FOCAL_GAMMA': 2.0, 'POS_WEIGHT': 1.0, 'NEG_WEIGHT': 1.0, 'FINE_TYPE': 'l2_with_std', 'FINE_WEIGHT': 1.0, 'FINE_CORRECT_THR': 1.0}
}, 

'DATASET': {'TRAINVAL_DATA_SOURCE': None, 'TRAIN_DATA_ROOT': None, 'TRAIN_POSE_ROOT': None, 'TRAIN_NPZ_ROOT': None, 'TRAIN_LIST_PATH': None, 'TRAIN_INTRINSIC_PATH': None, 'VAL_DATA_ROOT': None, 'VAL_POSE_ROOT': None, 'VAL_NPZ_ROOT': None, 'VAL_LIST_PATH': None, 'VAL_INTRINSIC_PATH': None, 'TEST_DATA_SOURCE': None, 'TEST_DATA_ROOT': None, 'TEST_POSE_ROOT': None, 'TEST_NPZ_ROOT': None, 'TEST_LIST_PATH': None, 'TEST_INTRINSIC_PATH': None, 'MIN_OVERLAP_SCORE_TRAIN': 0.4, 'MIN_OVERLAP_SCORE_TEST': 0.0, 'AUGMENTATION_TYPE': None, 'MGDPT_IMG_RESIZE': 640, 'MGDPT_IMG_PAD': True, 'MGDPT_DEPTH_PAD': True, 'MGDPT_DF': 8
}, 
'TRAINER': {'WORLD_SIZE': 1, 'CANONICAL_BS': 64, 'CANONICAL_LR': 0.006, 'SCALING': None, 'FIND_LR': False, 'OPTIMIZER': 'adamw', 'TRUE_LR': None, 'ADAM_DECAY': 0.0, 'ADAMW_DECAY': 0.1, 'WARMUP_TYPE': 'linear', 'WARMUP_RATIO': 0.0, 'WARMUP_STEP': 4800, 'SCHEDULER': 'MultiStepLR', 'SCHEDULER_INTERVAL': 'epoch', 'MSLR_MILESTONES': [3, 6, 9, 12], 'MSLR_GAMMA': 0.5, 'COSA_TMAX': 30, 'ELR_GAMMA': 0.999992, 'ENABLE_PLOTTING': True, 'N_VAL_PAIRS_TO_PLOT': 32, 'PLOT_MODE': 'evaluation', 'PLOT_MATCHES_ALPHA': 'dynamic', 'EPI_ERR_THR': 0.0005, 'POSE_GEO_MODEL': 'E', 'POSE_ESTIMATION_METHOD': 'RANSAC', 'RANSAC_PIXEL_THR': 0.5, 'RANSAC_CONF': 0.99999, 'RANSAC_MAX_ITERS': 10000, 'USE_MAGSACPP': False, 'DATA_SAMPLER': 'scene_balance', 'N_SAMPLES_PER_SUBSET': 200, 'SB_SUBSET_SAMPLE_REPLACEMENT': True, 'SB_SUBSET_SHUFFLE': True, 'SB_REPEAT': 1, 'RDM_REPLACEMENT': True, 'RDM_NUM_SAMPLES': None, 'GRADIENT_CLIPPING': 0.5, 'SEED': 66
}
}

_config = {'loftr': {'backbone_type': 'ResNetFPN', 
            'resolution': (8, 2), 
            'fine_window_size': 5, 
            'fine_concat_coarse_feat': True, 
            'resnetfpn': {'initial_dim': 128, 'block_dims': [128,196,256]}, 
            'coarse': {'d_model': 256, 'd_ffn': 256, 'nhead': 8, 'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'], 'attention': 'linear', 'temp_bug_fix': True}, 
            'match_coarse': {'thr': 0.2, 'border_rm': 2, 'match_type': 'dual_softmax', 'dsmax_temperature': 0.1, 'skh_iters': 3, 'skh_init_bin_score': 1.0, 'skh_prefilter': True, 'train_coarse_percent': 0.4, 'train_pad_num_gt_min': 200}, 
            'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self','cross'], 'attention': 'linear'}}}
# 'dataset': {'trainval_data_source': 'ScanNet', 'train_data_root': 'data/scannet/train', 'train_pose_root': None, 'train_npz_root': 'data/scannet/index/s...data/train', 'train_list_path': 'data/scannet/index/s...et_all.txt', 'train_intrinsic_path': 'data/scannet/index/i...insics.npz', 'val_data_root': 'data/scannet/test', 'val_pose_root': None, 'val_npz_root': 'assets/scannet_test_1500', ...}, 
# 'trainer': {'world_size': 4, 'canonical_bs': 64, 'canonical_lr': 0.006, 'scaling': 0.0625, 'find_lr': False, 'optimizer': 'adamw', 'true_lr': 0.000375, 'adam_decay': 0.0, 'adamw_decay': 0.1, ...}}






optimizer_1=tf.keras.optimizer.Adam(learning_rate=0.001)

matcher=LoFTR(config=_config['loftr']) # to be fixed
loss=LoFTRLoss(_config) # to be fixed

@tf.function
def train_step(data):
    '''
    data is a dictionary containing
    '''
    with tf.GradientTape() as tape:
        data = compute_supervision_coarse(data)
        data = matcher(data, training = True)
        data = compute_supervision_fine(data)
        data = loss(data)
    
    grads = tape.gradient(data['loss'], matcher.trainable_weights) # to be fixed
    optimizer_1.apply_gradients(zip(grads, matcher.trainable_weights))

    return data['loss']

epochs=100
batches = read_data('./Training/Ready','./Training/Ready')
loss_all=[]

for epoch in range(epochs):
    loss=0
    for batch in range(batches):
        loss+=train_step(batch)
    loss=tf.math.reduce_sum(loss)/(batches)
    loss_all.append(loss)















