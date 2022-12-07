import cv2 as cv
import tensorflow as tf
import h5py
import matplotlib.cm as cm
from src.loftr.utils.plotting_TF import make_matching_figure
from src.configs.getConfig import giveConfig
from src.loftr.LoFTR_TF import LoFTR

print(tf.__version__)
print(tf.keras.__version__)
config,_config = giveConfig()



#Creating the matcher 
matcher=LoFTR(config=_config['loftr']) 
matcher.load_weights("./weights/other/cp_smallMegadepth.ckpt")

#loading in the images for the current batch
img0_pth = "./other/capitolTest1.jpg"#"./src/training/datasets/unused/Scenes/scene3/Images/628.jpg"
img1_pth = "./other/capitolTest2.jpg"
img0_raw = cv.imread(img0_pth, cv.IMREAD_GRAYSCALE)
img1_raw = cv.imread(img1_pth, cv.IMREAD_GRAYSCALE)
img0_raw = cv.resize(img0_raw, (640, 480))
img1_raw = cv.resize(img1_raw, (640, 480))

img0 = tf.convert_to_tensor(img0_raw)[None][None]/255
img1 = tf.convert_to_tensor(img1_raw)[None][None]/255

data = {'image0': img0, 'image1': img1}

#Calling the matcher on the current batch
updata = matcher(data)

#Extracting matches
mkpts0 = updata['mkpts0_f'].numpy()
mkpts1 = updata['mkpts1_f'].numpy()
mconf = updata['mconf'].numpy()

color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path='./outputs/figs/matches.jpg')

print("DONE")



