import os
import sys
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.cm as cm
from loguru import logger
from tqdm import tqdm
print(os.getcwd())
from time import time
try:
  os.chdir("LoFTR-in-Tensorflow")
except:
  print("Directory is fine")

from src.loftr.LoFTR_TF import LoFTR
from src.training.supervisionTF import compute_supervision_coarse, compute_supervision_fine
from src.training.loftr_lossTF import LoFTRLoss
from src.training.dataloaders.megadepth.loadMD import read_data
from src.loftr.utils.plotting_TF import make_matching_figure
from src.configs.getConfig import giveConfig
# tf.config.run_functions_eagerly(True)

class trainer():
    def __init__(self):
        self.config,self._config = giveConfig()
        self.runningLoss = []
        self.dataDict = {}
        self.learning_rate = 0.0001
        self.warmupMultiplier = 0.0003
        self.A_optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.matcher=LoFTR(config=self._config['loftr']) 
        self.modelLoss=LoFTRLoss(self._config) 
         
    def getNewestData(self):
        return self.dataDict

    def saveWeights(self,checkpointPath):
        self.matcher.save_weights(checkpointPath)

    def loadWeights(self,checkpointPath):
        self.matcher.load_weights(checkpointPath)

    def train_step(self, input, epoch):
        '''
        data is a dictionary containing
        '''
        data = input

        with tf.GradientTape() as tape:
            superVisionData = compute_supervision_coarse(data,self.config)#Ground Truth generation
            modelData = self.matcher(superVisionData, training = True)
            fineSuperData = compute_supervision_fine(modelData,self.config)
            lossData = self.modelLoss(fineSuperData)

        grads = tape.gradient(lossData['loss'], self.matcher.trainable_weights, unconnected_gradients='zero')
        self.A_optimizer.apply_gradients(zip(grads, self.matcher.trainable_weights))
        # print("Weights Updated")

        #Train with changing learning rate
        # if (epoch+1) <= 3:
        #     self.learning_rate += self.warmupMultiplier
        #     self.A_optimizer.learning_rate.assign(self.learning_rate)
        if (epoch+1)%8==0:
            self.learning_rate /= 2
            self.A_optimizer.learning_rate.assign(self.learning_rate)

        return lossData['loss'],lossData


    def singleTest(self,imagePaths, outPath):
        img0_raw = cv.resize(cv.imread(imagePaths[0], cv.IMREAD_GRAYSCALE), (640, 480))
        img1_raw = cv.resize(cv.imread(imagePaths[1], cv.IMREAD_GRAYSCALE), (640, 480))
        img0 = tf.convert_to_tensor(img0_raw)[None][None]/255
        img1 = tf.convert_to_tensor(img1_raw)[None][None]/255
        data = {'image0': img0, 'image1': img1}

        #Calling the matcher on the current batch
        updata = self.matcher(data)

        mkpts0 = updata['mkpts0_f'].numpy()
        mkpts1 = updata['mkpts1_f'].numpy()
        mconf = updata['mconf'].numpy()

        color = cm.jet(mconf)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path=outPath)



def train(train_ds, trainer, epoch: int):
  epochLoss = 0.0
  for currentBatch in tqdm(train_ds,desc='Running Epoch '+str(epoch+1)):
    result,_ = trainer.train_step(currentBatch,epoch)
    epochLoss += (result)
  epochLoss = float(tf.math.reduce_sum(epochLoss)/(len(train_ds)))
  return epochLoss


def main(epochs,loadWeights=False):

    #Initialize Data Scenes summary helper
    t1 = time()
    scenes = read_data(batch_size=4)
    t2 = time()
    logger.info(f"Data Loaded {len(scenes)} scenes in {t2-t1} seconds")

    # scenes = strategy.experimental_distribute_dataset(scenes)
    myTrainer = trainer()

    if loadWeights:
        try:
            myTrainer.restoreCheck("./weights/other/cp_other.ckpt")
            myTrainer.loadWeights("./weights/other/cp_other.ckpt")
            logger.info(f'Weights Loaded!')
        except:
            logger.warning(f'No previous weights to load!')

    #Begin Training
    allLoss = []
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1:03d}/{epochs:03d}')

        start = time()
        currentLoss = train(scenes, myTrainer, epoch)
        logger.info(f'Current Loss = {currentLoss}')
        allLoss.append(currentLoss)
        end = time()
        logger.info(f'Time taken for Epoch {epoch+1} = {end-start}')

        myTrainer.saveWeights("./weights/test/cp_test.ckpt")
    print(allLoss)
    

    myTrainer.singleTest(["./other/scene0738_00_frame-000885.jpg",
    "./other/scene0738_00_frame-001065.jpg"],"./src/training/figs/matches_general.jpg")

if __name__ == '__main__':
#   main(parser.parse_args())
    main(10)
