from multiprocessing import Process, Manager
import struct
import zlib
import cv2
import tensorflow as tf
import glob
import imageio.v2 as imageio
import numpy as np
from loguru import logger
import time
from tqdm import tqdm
import os
# see https://github.com/open-mmlab/mmdetection3d/blob/v0.18.0/data/scannet/extract_posed_images.py for more details

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}

COMPRESSION_TYPE_DEPTH = {
    -1: 'unknown',
    0: 'raw_ushort',
    1: 'zlib_ushort',
    2: 'occi_ushort'
}


class RGBDFrame:
    """Class for single ScanNet RGB-D image processing."""

    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)),
            dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(
            struct.unpack('c' * self.color_size_bytes,
                          file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(
            struct.unpack('c' * self.depth_size_bytes,
                          file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        assert compression_type == 'zlib_ushort'
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        assert compression_type == 'jpeg'
        return imageio.imread(self.color_data)


class SensorData:
    """Class for single ScanNet scene processing.
    Single scene file contains multiple RGB-D images.
    """

    def __init__(self, filename):
        self.version = 4
        self.load(filename)
    
    def __del__(self):
        print('destroyed')

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(
                struct.unpack('c' * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack(
                'i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack(
                'i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                # if i in index:
                self.frames.append(frame)

    def export_depth_images(self, idx):
        depth_data = self.frames[idx].decompress_depth(
                self.depth_compression_type)
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
        tf_depth = tf.convert_to_tensor(depth,tf.float32)[None]
        return tf_depth

    def export_gray_images(self, idx):
        color = self.frames[idx].decompress_color(self.color_compression_type)
        gray= cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (640,480))
        tf_gray = tf.convert_to_tensor(gray, dtype=tf.float32)[None,None] / 255
        return tf_gray

    def export_poses(self, idx):
        pframe = self.frames[idx].camera_to_world
        tf_pose = tf.convert_to_tensor(pframe, dtype=tf.float32)
        return tf_pose

def extraction(sens_file_name, npz_path, sens_folders_path, intrinsics_path, score_thresh, no_sample_pairs_per_scene, batch_size, list_of_batches):
    """
    Output is a list of dictionaries

    Args:
    npz_path: path to folder of npz files (scannet_indices)
    sens_folders_path: path to folder of scene folders holding .sens files
    intrinsics_path: path to intrinsics file provided by loftr
    score_thresh: covisibility score threshold (0.4)
    no_sample_pairs_per_scene: total number of pairs sampled for a given scene
    batch_size: number of image pairs concatenated for training
    no_scenes: number of scenes to subsample for a given epoch

    Returns:
    List of Dicts with elements:
    {
        image0 (tf.tensor): (N,1,H,W)
        image1 (tf.tensor): (N,1,H,W)
        depth0 (tf.tensor): (N,H,W)
        depth1 (tf.tensor): (N,H,W)
        T_0to1 (tf.tensor): (N,4,4)
        T_1to0 (tf.tensor): (N,4,4)
        K0 (tf.tensor): (N,3,3)
        K1 (tf.tensor): (N,3,3)
    }
        
    """
    intrinsics_dict = np.load(intrinsics_path) #ONLY 1 INTRINSICS FILE FOR WHOLE DATASET
    # ctr = 0
    # for pair_info_filename_path in tqdm(glob.iglob(npz_path+'*'+'.npz')): #ITERATE FOR ALL NPZ FILES IN SCANNET_INDICES
    for pair_info_filename_path in tqdm(glob.iglob(npz_path+sens_file_name+'.npz')):
        scene_num = pair_info_filename_path.split(npz_path)[-1].split('.npz')[0] #EXTRACT SCENE NUMBER FROM FILENAME
        # stime = time.time()
        scene_pair_info_data = np.load(pair_info_filename_path) #GET THE SCENE INFO FROM THE NPZ FILE
        given_scene_pair_info, given_scene_pair_scores = scene_pair_info_data['name'], scene_pair_info_data['score']
        corrected_pair_info = [] #PAIRS WITH COVIS > THRESHOLD
        for idx in range(given_scene_pair_scores.shape[0]):
            if given_scene_pair_scores[idx] > score_thresh: #CHECK FOR COVISIBILITY SCORE ABOVE A THRESHOLD (PASS IN 0.4)
                corrected_pair_info.append(given_scene_pair_info[idx,:])
        rand_idxes = np.random.randint(len(corrected_pair_info), size=no_sample_pairs_per_scene) #RANDOMLY SAMPLE X PAIRS PER SCENE
        print(type(rand_idxes))
        print(f"{rand_idxes=}")
        
        data = SensorData(sens_folders_path+scene_num+'/'+scene_num+'.sens') #OPEN THE .SENS FILE AND READ BINARY INTO OBJECT
        K = tf.convert_to_tensor(intrinsics_dict[scene_num].reshape(3,3))[None] #INTRINSICS DO NOT CHANGE BETWEEN SCENES
        for rand_pair in rand_idxes:
            for i in range(batch_size):
                pair = corrected_pair_info[rand_pair][2:]
                pose0 = data.export_poses(pair[0])
                pose1 = data.export_poses(pair[1])
                if i == 0: #CREATE THE FIRST TENSOR
                    img0 = data.export_gray_images(pair[0])
                    img1 = data.export_gray_images(pair[1])
                    depth0 = data.export_depth_images(pair[0])
                    depth1 = data.export_depth_images(pair[1])
                    T0to1 = tf.linalg.matmul(pose1, tf.linalg.inv(pose0))
                    T1to0 = tf.linalg.inv(T0to1)[None]
                    T0to1 = T0to1[None]
                    Kout = K
                else: #CONCATENATE ABOVE TENSORS
                    img0 = tf.concat((img0, data.export_gray_images(pair[0])),axis=0)
                    img1 = tf.concat((img1, data.export_gray_images(pair[1])),axis=0)
                    depth0 = tf.concat((depth0, data.export_depth_images(pair[0])),axis=0)
                    depth1 = tf.concat((depth1, data.export_depth_images(pair[1])),axis=0)
                    temp_T0to1 = tf.linalg.matmul(pose1, tf.linalg.inv(pose0))
                    temp_T1to0 = tf.linalg.inv(temp_T0to1)[None]
                    temp_T0to1 = temp_T0to1[None]
                    T0to1 = tf.concat((T0to1, temp_T0to1),axis=0)
                    T1to0 = tf.concat((T1to0, temp_T1to0),axis=0)   
                    Kout = tf.concat((Kout, K), axis=0)        
            list_of_batches.append({'image0':img0, 'depth0':depth0, 'T_0to1':T0to1, 'T_1to0':T1to0, 'K0':Kout, 'K1':Kout, 'image1':img1, 'depth1':depth1}) #BUILD DATASET AS LIST OF DICTS
        del data

        # msg = 'loop num ' + str(ctr) + ' ; ' + scene_num + ' loop time ' + str(round(time.time()-stime, 1))
        # ctr +=1
        # logger.info(f'scene finished: {msg}')    
    return list_of_batches
    
def import_scannet(npz_path, sens_folders_path, intrinsics_path, score_thresh, no_sample_pairs_per_scene, batch_size, no_scenes):
    temp_sens_files_list = os.listdir(sens_folders_path)
    rand_files = np.random.randint(len(temp_sens_files_list), size=no_scenes)
    sens_files_list = []
    for num in rand_files:
        sens_files_list.append(temp_sens_files_list[num])
    print(f"{sens_files_list=}")
    start = time.time()
    with Manager() as manager:
        list_of_batches = manager.list()  # <-- can be shared between processes.   #OUTPUT = LIST OF DICTS
        processes = []
        for sens_file_name in sens_files_list: 
            p = Process(target=extraction, args=(sens_file_name, npz_path, sens_folders_path, intrinsics_path, score_thresh, no_sample_pairs_per_scene, batch_size, list_of_batches))  # Passing the list ot the processes
            p.start()
            processes.append(p)
        for p in processes:
            p.join()  #BUILD DATASET AS LIST OF DICTS
            list_of_batches = list(list_of_batches)
    end = time.time() 
    print("time taken for "+ str(no_scenes) + " scenes =" + str(end-start) + "seconds")

    # SAVING THE LIST OF DICTIONARIES of TENSORS AS A ".npy" FILE
    np.save( "/Users/surajreddy/Desktop/perception project/list_test", list_of_batches)

    return list_of_batches

# if __name__ == "__main__":
   
    # batch_list = import_scannet('/Users/surajreddy/Downloads/scannet_indices 2/scene_data/train/', '/Users/surajreddy/Desktop/perception project/sens_files/', '/Users/surajreddy/Downloads/scannet_indices 2/intrinsics.npz', 0.4, 40, 4, 5)
    # print(f"{batch_list=}")
    # print(f"{len(batch_list)}")
    # print(f"{batch_list[1]=}")

