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
# os.chdir("LoFTR-in-Tensorflow")

from src.loftr.LoFTR_TF import LoFTR
from src.training.supervisionTF import compute_supervision_coarse, compute_supervision_fine
from src.training.loftr_lossTF import LoFTRLoss
from src.training.datasets.load_scannet import import_scannet

from src.loftr.utils.plotting_TF import make_matching_figure
from src.configs.getConfig import giveConfig
# tf.config.run_functions_eagerly(True)

class trainer():
    def __init__(self):
        self.config,self._config = giveConfig()
        self.runningLoss = []
        self.learning_rate = 6e-3
        self.A_optimizer=tf.keras.optimizers.Adam(learning_rate=6e-4)
        self.matcher=LoFTR(config=self._config['loftr']) 
        self.modelLoss=LoFTRLoss(self._config) 
         

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

        if (epoch+1)%3==0 and epoch+1<=30:
            self.learning_rate/=2
            self.A_optimizer.learning_rate.assign(self.learning_rate)
        # print("Weights Updated")

        return lossData['loss']

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
  for currentBatch in tqdm(train_ds,desc='Running Epoch '+str(epoch+ 1)):
    result = trainer.train_step(currentBatch, epoch)
    # logger.info(f'running...')
    epochLoss+= result
  
  epochLoss = float(tf.math.reduce_sum(epochLoss)/(len(train_ds)))
  return epochLoss


def main(epochs):

    # initialize tf.distribute.MirroredStrategy
    # args.global_batch_size = num_devices * args.batch_size

    # initialize TensorBoard summary helper
    t1 = time()
    npz_dir = './src/training/datasets/scannet/train/'
    sens_dir = './src/training/datasets/scannet/scans/'
    intrins = './src/training/datasets/scannet/intrinsics.npz'
    scenes = import_scannet(npz_dir,sens_dir, intrins, 0.4, 64, 8, 5)
    # logger.info(scenes)
    t2 = time()
    logger.info(f"Data Loaded {len(scenes)} batches in {(t2-t1)/60} minutes")

    # scenes = strategy.experimental_distribute_dataset(scenes)

    myTrainer = trainer()
    # try:
    #     myTrainer.loadWeights("./weights/big_test/cp_SCANNET_big.ckpt")
    # except:
    #     logger.warning(f'No previous weights to load!')

    allLoss = []
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1:03d}/{epochs:03d}')
        start = time()
        currentLoss = train(scenes, myTrainer, epoch)
        logger.info(f'Current Loss = {currentLoss}')
        allLoss.append(currentLoss)
        # results = test(args, test_ds, gan, summary, epoch)
        end = time()
        logger.info(f'Time taken for Epoch {epoch+1} = {(end-start)/60} minutes')

        # if epoch % 10 == 0:
        # gan.save_checkpoint()
        # utils.plot_cycle(plot_ds, gan, summary, epoch)
        myTrainer.saveWeights("./weights/undistort_test/scannet_undistorted.ckpt")
    print(allLoss)
    

    myTrainer.singleTest(["./other/scene0738_00_frame-000885.jpg",
    "./other/scene0738_00_frame-001065.jpg"],"./src/training/figs/matches_miniSCANNET_testing.jpg")

if __name__ == '__main__':
#   main(parser.parse_args())
    main(100)
