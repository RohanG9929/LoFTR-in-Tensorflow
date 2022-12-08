import numpy
import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np
import scipy
from scipy.io import loadmat
import tensorflow as tf

def find_pairs():
    images_files=glob.glob(r'C:\Users\15512\Downloads\toolbox_nyu_depth_v2\library_0001a (Done_new)\images_undistorted\*.png')
    depth_files=glob.glob(r'C:\Users\15512\Downloads\toolbox_nyu_depth_v2\library_0001a (Done_new)\depth_projected\*.mat')
    pair_indices={}
    result=[]
    for i in range(len(images_files)):
        min_norm=float("inf")
        img_0_num=images_files[i].split('.')[0][-2:]
        img_0_num=int(img_0_num)
        print(f"image 0 number is {img_0_num}")
        print(f'type of image 0 num is {type(img_0_num)}')
        if str(i) in pair_indices.keys():
            continue
        if i==len(images_files)-1:
            break
        ########## read first image and convert to gray scale#########
        img_0=cv2.imread(images_files[i])
        img_0_gray=cv2.cvtColor(img_0,cv2.COLOR_BGR2GRAY)
        img_0_gray=img_0_gray.reshape((1,480,640))
        #############################################################
        for j in range(i+1,len(images_files)):
            if str(j) in pair_indices.keys():
                continue
            img_1_num=images_files[j].split('.')[0][-2:]
            img_1_num=int(img_1_num)
            img_1=cv2.imread(images_files[j])
            img_1_gray=cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
            img_1_gray=img_1_gray.reshape((1,480,640))
            ORB = cv2.ORB_create()
            kp_1, des_1 = ORB.detectAndCompute(img_0, None)
            kp_2, des_2 = ORB.detectAndCompute(img_1, None)
            norm=np.linalg.norm(des_1-des_2)
            if norm<min_norm and norm!=0:
                min_norm=norm
                matching_pairs=(img_0_gray,img_1_gray)
                depth0_file=r'C:\Users\15512\Downloads\toolbox_nyu_depth_v2\library_0001a (Done_new)\depth_projected\ {0}.mat'.format(str(img_0_num))
                print(f'depth0_file after reading is {depth0_file}')
                depth_zero=loadmat(depth0_file)
                depth1_file=r'C:\Users\15512\Downloads\toolbox_nyu_depth_v2\library_0001a (Done_new)\depth_projected\ {0}.mat'.format(str(img_1_num))
                depth_one=loadmat(depth1_file)
                #print(f'depth1_file is {depth1_file}')
                depth_pairs=(depth_zero['depthOut'].reshape((1,480,640)),depth_one['depthOut'].reshape((1,480,640)))
                index_i=str(i)
                index_i_val=i
                index_j=str(j)
                index_j_val=j
        result.append([matching_pairs,depth_pairs])
        pair_indices[index_i]=index_i_val
        pair_indices[index_j]=index_j_val
    return result,pair_indices

def save_images():
    '''
    save pairs of images for validation of overlapping region
    '''
    result,_=find_pairs()
    dir_images="images_test"
    parent_dir=r"C:\Users\15512\Downloads\toolbox_nyu_depth_v2\library_0001a (Done_new)"
    path_images = os.path.join(parent_dir, dir_images)
    os.mkdir(path_images)
    for i in range(len(result)):
        pair_0=result[i][0][0].reshape((480,640))
        pair_1=result[i][0][1].reshape((480,640))
        new_path_1=os.path.join(path_images,str(2*i))
        os.mkdir(new_path_1)
        new_path_2=os.path.join(path_images,str(2*i+1))
        os.mkdir(new_path_2)
        cv2.imwrite(os.path.join(new_path_1,'candidate1.jpg'),pair_0)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(new_path_2,'candidate2.jpg'),pair_1)
        cv2.waitKey(0)

def return_dic():
    '''
    returns a dictionary to work with tensors.
    '''
    k0=k1=np.array([[5.1885790117450188e+02,0,3.2558244941119034e+02],
        [0,5.1946961112127485e+02,2.5373616633400465e+02],[0,0,1]])
    k0=k1=k0.reshape((1,3,3))
    k0_final=k1_final=np.repeat(k0,4,axis=0)
    T_0to1=np.array([[[-1,0.0050,0.0043,2.5031875059141302e-02],
        [-0.0051,-1.0000, -0.0037,-2.9342312935846411e-04],[-0.0043,0.0037,-1.0000,6.6238747008330102e-04],[0,0,0,1]]])
    T_0to1_final= T_0to1_final=np.repeat(T_0to1,4,axis=0)
    result,_=find_pairs()
    final_dic=[]
    for i in range(0,len(result)-4,4):
        images_zero=np.concatenate((result[i][0][0],
                        result[i+1][0][0],result[i+2][0][0],result[i+3][0][0]),axis=0)
        images_zero=tf.convert_to_tensor(images_zero)
        depths_zero=np.concatenate((result[i][1][0],
                        result[i+1][1][0],result[i+2][1][0],result[i+3][1][0]),axis=0)
        depths_zero=tf.convert_to_tensor(depths_zero)
        images_one=np.concatenate((result[i][0][1],
                        result[i+1][0][1],result[i+2][0][1],result[i+3][0][1]),axis=0)
        images_one=tf.convert_to_tensor(images_one)
        depths_one=np.concatenate((result[i][1][1],
                        result[i+1][1][1],result[i+2][1][1],result[i+3][1][1]),axis=0)
        depths_one=tf.convert_to_tensor(depths_one)
        final_dic.append({"image0":images_zero,"image1":images_one,"depth0":depths_zero,"depth1":depths_one,
                            'K0':k0_final,"K1":k0_final,"T_0to1":T_0to1_final,"T_1to0":T_0to1_final})
    
    return final_dic

fin=return_dic()
np.save("library",fin)
print(len(fin))
print(fin[0]['depth0'].shape)
print(fin[0]['depth1'].shape)
print(fin[0]['image0'].shape)
print(fin[0]['image1'].shape)
print(type(fin[0]['image1']))

#sc1=np.load(r"C:\Users\15512\Desktop\cafe_c.npy",allow_pickle=True)
#print(sc1.shape)
#sc1_list=list(sc1)
#print(len(sc1_list))






        
            
