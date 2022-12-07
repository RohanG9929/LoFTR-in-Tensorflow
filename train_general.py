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
from src.training.datasets.LoadDataMD import read_fullMD_data
from src.training.datasets.loadMD import read_data

from src.loftr.utils.plotting_TF import make_matching_figure
from src.configs.getConfig import giveConfig
# tf.config.run_functions_eagerly(True)

class trainer():
    def __init__(self,num_devices, strategy: tf.distribute.Strategy):
        self.strategy = strategy
        self.config,self._config = giveConfig()
        self.num_devices = num_devices
        self.runningLoss = []

        with self.strategy.scope():
            self.A_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            self.matcher=LoFTR(config=self._config['loftr']) 
            self.modelLoss=LoFTRLoss(self._config) 
         
    def getNumDevices(self):
        return self.num_devices

    def saveWeights(self,checkpointPath):
        self.matcher.save_weights(checkpointPath)

    def loadWeights(self,checkpointPath):
        self.matcher.load_weights(checkpointPath)
        
    def train_step(self, input):
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

        return lossData['loss']

    # @tf.function
    def distributed_train_step(self, currentBatch):
        batchLoss = self.strategy.run(self.train_step, args=([currentBatch]))
        self.runningLoss.append(batchLoss)
        rbatchLoss = self.strategy.experimental_local_results(batchLoss)
        return rbatchLoss

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
  epochLoss = []
  for currentBatch in tqdm(train_ds,desc='Running Epoch '+str(epoch+ 1)):
    result = trainer.distributed_train_step(currentBatch)
    # logger.info(f'running...')
    for idx in range(trainer.getNumDevices()):
        epochLoss += (result[idx])

  epochLoss = float(tf.math.reduce_sum(epochLoss)/(len(train_ds)))
  return epochLoss


def main(epochs):
    tf.keras.backend.clear_session()

    np.random.seed(1234)
    tf.random.set_seed(1234)

    # initialize tf.distribute.MirroredStrategy
    strategy = tf.distribute.MirroredStrategy(devices=None,cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    num_devices = strategy.num_replicas_in_sync
    # args.global_batch_size = num_devices * args.batch_size
    logger.info(f'Number of devices: {num_devices}')

    # initialize TensorBoard summary helper
    t1 = time()
    scenes = read_data(batch_size=4)
    t2 = time()
    logger.info(f"Data Loaded {len(scenes)} scenes in {t2-t1} seconds")

    # scenes = strategy.experimental_distribute_dataset(scenes)

    myTrainer = trainer(num_devices,strategy=strategy)

    allLoss = []
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1:03d}/{epochs:03d}')

        start = time()
        currentLoss = train( scenes, myTrainer, epoch)
        logger.info(f'Current Loss = {currentLoss}')
        allLoss.append(currentLoss)
        # results = test(args, test_ds, gan, summary, epoch)
        end = time()
        logger.info(f'Time taken for Epoch {epoch+1} = {end-start}')

        # if epoch % 10 == 0:
        # gan.save_checkpoint()
        # utils.plot_cycle(plot_ds, gan, summary, epoch)
    print(allLoss)
    myTrainer.saveWeights("./weights/miniMegadepthStrat/cp_Megadepth.ckpt")

    myTrainer.singleTest(["./other/scene0738_00_frame-000885.jpg",
    "./other/scene0738_00_frame-001065.jpg"],"./src/training/figs/matches_miniMD.jpg")

if __name__ == '__main__':
#   main(parser.parse_args())
    main(30)
