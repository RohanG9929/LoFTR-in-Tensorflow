import numpy as np
import cv2
# import tensorflow as tf
import scipy
from scipy.io import loadmat
import os
import glob


def giveK():
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    K = np.array(([fx_rgb,0,cx_rgb],[0,fy_rgb,cy_rgb],[0,0,1]))

    return K

def giveT():

    R = -np.array([ 9.9997798940829263e-01, 5.0518419386157446e-03, 
    4.3011152014118693e-03, -5.0359919480810989e-03, 
    9.9998051861143999e-01, -3.6879781309514218e-03, 
    -4.3196624923060242e-03, 3.6662365748484798e-03, 
    9.9998394948385538e-01 ])

    R = R.reshape([3,3],order='F')
    R = np.linalg.inv(R.T)

    t_x = 2.5031875059141302e-02
    t_z = -2.9342312935846411e-04
    t_y = 6.6238747008330102e-04

    T0_to_1 = np.hstack((R,np.array([[t_x,t_y,t_z]]).T))
    T0_to_1 = np.vstack((T0_to_1,np.zeros(4)))
    T0_to_1[-1,-1] = 1
    # print(T0_to_1)


    T1_to_0 = np.hstack((R.T,-R.T@np.array([[t_x,t_y,t_z]]).T))
    T1_to_0 = np.vstack((T1_to_0,np.zeros(4)))
    T1_to_0[-1,-1] = 1
    # print(T1_to_0)

    return T0_to_1, T1_to_0

def read_data(path_imgs,path_depth):#,transf,intrinsics,no_of_pairs=5):
    '''
    returns a dictionary containing the data
     data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_left_to_right': T_0to1,  # (4, 4)
            'T_right_to_left': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
        }

    '''
    batches=[]
    ################## ead images ########################
    images=glob.glob(""+path_imgs+"/*.jpg")
    #################### read depths ######################
    depth=glob.glob(""+path_depth+"/*.mat")
    assert len(images)==len(depth)

    count=0

    #Order list
    orderedImages = []
    fileNums = []
    for file in images:
        fileNums.append(int(file.split('.')[1].split('/')[-1]))
    fileNums.sort()

    for num in fileNums:
        orderedImages.append(""+path_imgs+"/"+str(num)+".jpg")

    data={}
    for image in list(orderedImages):
        image_num=image.split('.')[1].split('/')[-1]
        currImage=cv2.imread(image)
        for d in list(depth):
            if (d.split('.')[1].split('/')[-1]==image_num):
                # read depth and increment count
                myDepth=loadmat(d)['depthOut']
                count+=1
                if count==1:
                    data['image0']=currImage
                    data['depth0']=myDepth
                    data['T0_to_1'] = giveT()[0]
                    data['T1_to_0'] = giveT()[1]
                    data['K_0'] = giveK()
                    data['K_1'] = giveK()
                    break
                elif count==2:
                    data['image1']=currImage
                    data['depth1']=myDepth
                    count=0
                    batches.append(data)
                    data={}
                    break


    return batches
            
                        

 
batches = read_data('./Training/Ready','./Training/Ready')
    
print("DONE")
    # ################## camera extrinisics###############
    # T_left_to_right=np.zeros((4,4))
    # T_right_to_left=np.zeros((4,4))
    # T_right_to_left[:3,:3]=np.transpose(T_left_to_right[:3,:3]) # inverse or transpose of rotation matrix
    # T_right_to_left[:3,3]=np.transpose(T_left_to_right[:3,:3])@T_left_to_right[:3,3]
    # ##################### intrinsics#######################
    # k0=np.zeros((3,3))
    # k1=np.zeros((3,3))

    # ############ update 



    



