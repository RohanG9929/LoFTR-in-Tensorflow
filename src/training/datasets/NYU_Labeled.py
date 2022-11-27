import numpy as np
import h5py
import cv2
import random

# data path
path_to_depth = r'/scratch/fsa4859/loftr_tensorflow/nyu_depth_v2_labeled.mat'

# read mat file
f = h5py.File(path_to_depth)

print('the length of dataset')
print(len(f['images']))
l=len(f['images'])

results=[]
for i in range(l):
    image=f['images'][i]
    image=image.astype('float32')
    img_ = np.zeros([480, 640, 3])
    img_=img_.astype('float32')
    img_[:,:,0] = image[0,:,:].T
    img_[:,:,1] = image[1,:,:].T
    img_[:,:,2] = image[2,:,:].T
    depth = f['depths'][i]
    depth_ = np.empty([480, 640])
    depth_[:,0] = depth[0,:].T
    depth_[:,1] = depth[1,:].T
    results.append({'image':img_,"depth":depth_})

no_pairs=700
data=[]
np_batches=0
image_concatenated_zero=np.zeros((4,1,480,640))
image_concatenated_one=np.zeros((4,1,480,640))
depth_zero=np.zeros((4,480,640))
depth_one=np.zeros((4,480,640))
for pair in range(no_pairs):
    i1=random.randint(0,1448)
    print(f'first index is {i1}')
    i2=random.randint(0,1448)
    print(f'second index is {i2}')
    image_cand1=results[i1]['image']
    depth_cand1=results[i1]['depth']
    print(f'the shape of image candidate is {image_cand1.shape}')
    print(f'type of image 1 is{type(image_cand1)}')
    image_cand2=results[i2]['image']
    depth_cand2=results[i2]['depth']
    print(f'the shape of image candidate 2 is {image_cand2.shape}')
    image_cand1_gray=cv2.cvtColor(image_cand1,cv2.COLOR_BGR2GRAY)
    print(f'the shape of image after gray scale conversion is {image_cand1_gray.shape}')
    print(f'dtype of image is {image_cand1_gray.dtype}')
    image_cand2_gray=cv2.cvtColor(image_cand2,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp_1, des_1 = orb.detectAndCompute(image_cand1_gray.astype('uint8'), None)
    print(f'length of keypoints is {len(kp_1)}')
    kp_2, des_2 = orb.detectAndCompute(image_cand2_gray.astype('uint8'), None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_1,des_2)
    matches = sorted(matches, key = lambda x:x.distance)
    perc_overlap=len(matches)/len(kp_1)
    #perc_overlap=len(kp_1)/len(matches)
    print(f'the percentage overlap is {perc_overlap}')
    if perc_overlap>=0.25:
        np_batches+=1
        if np_batches<4:
            image_concatenated_zero[np_batches,:,:,:]=image_cand1_gray.reshape(1,1,480,640)
            image_concatenated_one[np_batches,:,:,:]=image_cand2_gray.reshape(1,1,480,640)
            depth_zero[np_batches,:,:]=depth_cand1.reshape(1,480,640)
            depth_one[np_batches,:,:]=depth_cand2.reshape(1,480,640)
            continue
        data.append({'image0':image_concatenated_zero,'depth0':depth_zero,'depth1':depth_one,'image1':image_concatenated_one})


print(data[0]['image0'].shape)
print(data[0]['image1'].shape)
print(data[0]['depth0'].shape)
print(data[0]['depth1'].shape)
print(len(data))
file='labeled_data_nyu.npy'
data_np=np.array(data)
np.save(file,data_np)





