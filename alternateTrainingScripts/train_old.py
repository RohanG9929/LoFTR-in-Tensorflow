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
from src.training.datasets.loadMD import read_data
from src.loftr.utils.plotting_TF import make_matching_figure
from src.configs.getConfig import giveConfig
tf.config.run_functions_eagerly(True)

config,_config = giveConfig()
checkpointPath = "./weights/cp_smallMegadepth.ckpt"

optimizer_1=tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer_2 = tf.keras.optimizers.experimental.AdamW()
# train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
# val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

matcher=LoFTR(config=_config['loftr']) 
modelLoss=LoFTRLoss(_config) 

# matcher.compile(optimizer=optimizer_1,
#                 loss=modelLoss)
##############################
#Init Training
##############################
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
    # print("Weights Updated")

    return lossData['loss']


epochs = 5
scenes = read_data(batch_size=4)
logger.info(f"Data Loaded!")
loss_all=[]
logger.info(f"Trainer initialized!")

##############################
#Begin Training
##############################
for epoch in range(epochs):
    loss=0
    for batch in tqdm((scenes),desc='Running Epoch '+str(epoch)):
        loss+=train_step(batch)
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
img0_pth = "./other/scene0738_00_frame-000885.jpg"#"./src/Training/Scenes/scene1/Images/603.jpg"
img1_pth = "./other/scene0738_00_frame-001065.jpg"#"./src/Training/Scenes/scene1/Images/604.jpg"
img0_raw = cv.resize(cv.imread(img0_pth, cv.IMREAD_GRAYSCALE), (640, 480))
img1_raw = cv.resize(cv.imread(img1_pth, cv.IMREAD_GRAYSCALE), (640, 480))

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














