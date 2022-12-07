# 1- define optimizer
# 2- one step through optimizer
# 3- train val inference
# 4-  training_step
# 5- training epoch end
# 6- compute metrics

import os
import sys
import tensorflow as tf
import cv2 as cv
import matplotlib.cm as cm
from loguru import logger
from tqdm import tqdm
print(os.getcwd())
# os.chdir("LoFTR-in-Tensorflow")

from src.loftr.LoFTR_TF import LoFTR
from src.training.supervisionTF import compute_supervision_coarse, compute_supervision_fine
from src.training.loftr_lossTF import LoFTRLoss
from src.training.datasets.LoadDataMD import read_fullMD_data
from src.loftr.utils.plotting_TF import make_matching_figure, make_matching_figures
from src.configs.getConfig import giveConfig
from src.training.metricsTF import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)

tf.config.run_functions_eagerly(True)

config,_config = giveConfig()
checkpointPath = "./weights/fullMD/cp_smallMegadepth.ckpt"

optimizer_1=tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer_2 = tf.keras.optimizers.experimental.AdamW()
# train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
# val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

matcher=LoFTR(config=_config['loftr']) 
modelLoss=LoFTRLoss(_config) 



##############################
#Init Training Functinos
##############################

def _compute_metrics(self, batch):

    compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
    compute_pose_errors(batch, config)  # compute R_errs, t_errs, pose_errs for each pair

    rel_pair_names = list(zip(*batch['pair_names']))
    bs = batch['image0'].shape[0]
    metrics = {
        # to filter duplicate pairs caused by DistributedSampler
        'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
        'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
        'R_errs': batch['R_errs'],
        't_errs': batch['t_errs'],
        'inliers': batch['inliers']}
    ret_dict = {'metrics': metrics}
    return ret_dict, rel_pair_names

@tf.function
def train_step(data):
    '''
    data is a dictionary containing
    '''
    with tf.GradientTape() as tape:
        superVisionData = compute_supervision_coarse(data,config)#Ground Truth generation
        modelData = matcher(superVisionData, training = True)
        fineSuperData = compute_supervision_fine(modelData,config)
        lossData = modelLoss(fineSuperData)

    grads = tape.gradient(lossData['loss'], matcher.trainable_weights, unconnected_gradients='zero')
    optimizer_1.apply_gradients(zip(grads, matcher.trainable_weights))

    return lossData['loss']

@tf.function
def validation_step(data):
    superVisionData = compute_supervision_coarse(data,config)#Ground Truth generation
    modelData = matcher(superVisionData, training = True)
    fineSuperData = compute_supervision_fine(modelData,config)
    batch = modelLoss(fineSuperData)
    ret_dict, _ = _compute_metrics(batch)
    
    val_plot_interval = 1#max(numValBatches // self.n_vals_plot, 1)
    figures = {'evaluation': []}
    # if batch_idx % val_plot_interval == 0:
    figures = make_matching_figures(batch, config)

    return {
        **ret_dict,
        'loss_scalars': batch['loss_scalars'],
        'figures': figures,
    }

root_dir = './src/training/datasets/megadepth/'
epochs = 10
scenes = read_fullMD_data(batch_size=4,npz_dir= os.path.join(root_dir,'megadepth_indices/scene_info_0.1_0.7/*'),root_dir=root_dir)
validation_scenes = []#read validation data 
logger.info(f"Data Loaded!")
loss_all=[]
validation_all=[]
logger.info(f"Trainer initialized!")

##############################
#Begin Training
##############################
for epoch in range(epochs):
    loss=0
    #Epoch begin
    for batch in tqdm((scenes),desc='Running Epoch '+str(epoch)):
        loss+=train_step(batch)
    
    #Epoch end    
    validation_all.append(validation_step(validation_scenes[epoch]))
    print(f'loss for epoch {epoch} is {tf.math.reduce_sum(loss)/(len(scenes))}')
    loss_all.append(float(tf.math.reduce_sum(loss)/(len(scenes))))


######################################################################mess
logger.info(f"Training Done!")
matcher.save_weights(checkpointPath)

print("Loss progression is:")
print(loss_all)
print('')

matcher.summary()
tf.config.run_functions_eagerly(False)
######################################################################

#loading in the images for the current batch
img0_raw = cv.resize(cv.imread("./other/scene0738_00_frame-000885.jpg", cv.IMREAD_GRAYSCALE), (640, 480))
img1_raw = cv.resize(cv.imread("./other/scene0738_00_frame-001065.jpg", cv.IMREAD_GRAYSCALE), (640, 480))
img0 = tf.convert_to_tensor(img0_raw)[None][None]/255
img1 = tf.convert_to_tensor(img1_raw)[None][None]/255
data = {'image0': img0, 'image1': img1}

#Calling the matcher on the current batch
updata = matcher(data)

mkpts0 = updata['mkpts0_f'].numpy()
mkpts1 = updata['mkpts1_f'].numpy()
mconf = updata['mconf'].numpy()

color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path='./src/training/figs/matches.jpg')

print("DONE")














