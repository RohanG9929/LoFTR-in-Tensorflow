import numpy as np
import h5py
import cv2
import io
import tensorflow as tf
import os.path as osp
from tqdm import tqdm
import glob

root_dir = './Training/megadepth/' # This should be the phoenix/.../megadepth pathway

img_resize=640#None
df=None
img_padding=True
depth_padding=True
augment_fn=None
depth_max_size = 2000 if depth_padding else None

# self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
coarse_scale = 0.125

def load_array_from_s3(
    path, client, cv_type,
    use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data



def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = cv2.imread(path,0)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    # image = cv2.resize(image, (480, 640))
    scale = tf.convert_to_tensor([w/w_new, h/h_new], dtype=tf.double)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
        mask = tf.convert_to_tensor(mask)
    else:
        mask = None

    image = tf.convert_to_tensor(image,tf.float32)[None] / 255  # (h, w) -> (1, h, w) and normalized
    

    return image, mask, scale


def read_megadepth_depth(path, pad_to=None):
    # if str(path).startswith('s3://'):
    #     depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    # else:
    depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = tf.convert_to_tensor(depth,tf.float32)  # (h, w)
    return depth

def loadMD(data,idx):
        (idx0, idx1), overlap_score, central_matches = data['pair_infos'][idx]
        image0_path_name = data['depth_paths'][idx0].replace('depths', 'imgs').replace('.h5', '.jpg')
        image1_path_name = data['depth_paths'][idx1].replace('depths', 'imgs').replace('.h5', '.jpg')
        # phoenix/.../MDv1/scene_no/dense_folder/
        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(root_dir, image0_path_name)
        img_name1 = osp.join(root_dir, image1_path_name)
    


        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0 = read_megadepth_gray(img_name0, img_resize, df, img_padding, None)
            # np.random.choice([augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(img_name1, img_resize, df, img_padding, None)
            # np.random.choice([augment_fn, None], p=[0.5, 0.5]))

        # read depth. shape: (h, w)

        depth0 = read_megadepth_depth(
            osp.join(root_dir, data['depth_paths'][idx0]), pad_to=depth_max_size)
        depth1 = read_megadepth_depth(
            osp.join(root_dir, data['depth_paths'][idx1]), pad_to=depth_max_size)


        # read intrinsics of original size
        K_0 = tf.convert_to_tensor(data['intrinsics'][idx0].copy(), dtype=tf.float32)#.reshape(3, 3)
        K_1 = tf.convert_to_tensor(data['intrinsics'][idx1].copy(), dtype=tf.float32)#.reshape(3, 3)

        # read and compute relative poses
        T0 = data['poses'][idx0]
        T1 = data['poses'][idx1]
        T_0to1 = tf.convert_to_tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=tf.float32)[:4, :4]  # (4, 4)
        T_1to0 = tf.linalg.inv(T_0to1)#.inverse()

        outdata = {
            'image0': tf.reshape(image0,[1,1,image0.shape[1],image0.shape[2]]),  # (1, h, w)
            'depth0': tf.reshape(depth0,[1,depth0.shape[0],depth0.shape[1]]),  # (h, w)
            'image1': tf.reshape(image1,[1,1,image1.shape[1],image1.shape[2]]),
            'depth1': tf.reshape(depth1,[1,depth1.shape[0],depth1.shape[1]]),
            'T_0to1': tf.reshape(T_0to1,[1,T_0to1.shape[0],T_0to1.shape[1]]),  # (4, 4)
            'T_1to0': tf.reshape(T_1to0,[1,T_1to0.shape[0],T_1to0.shape[1]]),
            'K0': tf.reshape(K_0,[1,K_0.shape[0],K_0.shape[1]]),  # (3, 3)
            'K1': tf.reshape(K_1,[1,K_1.shape[0],K_1.shape[1]]),
            'scale0': tf.reshape(scale0,[1,scale0.shape[0]]),  # [scale_w, scale_h]
            'scale1': tf.reshape(scale1,[1,scale1.shape[0]]),
            # 'dataset_name': 'MegaDepth',
            # 'scene_id': scene_id,
            # 'pair_id': idx,
            # 'pair_names': (data['image_paths'][idx0], data['image_paths'][idx1]),
        }

        # # for LoFTR training
        # if mask0 is not None:  # img_padding is True
        #     if coarse_scale:
        #         [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
        #                                                scale_factor=coarse_scale,
        #                                                mode='nearest',
        #                                                recompute_scale_factor=False)[0].bool()
        #     data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return outdata



def read_data(batch_size, npz_dir):
    list_of_scenes_by_covisibility_score = []
    reduce_data_size = 10
    for npz_file in glob.glob(npz_dir):
        data = np.load(npz_file,allow_pickle=True)
    
        scene_by_covisibility_score=[]

        for i in tqdm(range(int(len(data['pair_infos'])/reduce_data_size)),ascii=True,desc='Loading Scenes'):
            if i==0 or len(finalData)==0:
                finalData = loadMD(data,i)
            else:
                newData = loadMD(data,i)
                finalData['image0'] = tf.concat((finalData['image0'],newData['image0']),axis=0)
                finalData['depth0'] = tf.concat((finalData['depth0'],newData['depth0']),axis=0)
                finalData['T_0to1'] = tf.concat((finalData['T_0to1'],newData['T_0to1']),axis=0)
                finalData['T_1to0'] =  tf.concat((finalData['T_1to0'],newData['T_1to0']),axis=0)
                finalData['K0'] = tf.concat((finalData['K0'],newData['K0']),axis=0)
                finalData['K1'] =  tf.concat((finalData['K1'],newData['K1']),axis=0)
                finalData['image1'] = tf.concat((finalData['image1'],newData['image1']),axis=0)
                finalData['depth1'] = tf.concat((finalData['depth1'],newData['depth1']),axis=0)
                finalData['scale0'] = tf.concat((finalData['scale0'],newData['scale0']),axis=0)
                finalData['scale1'] = tf.concat((finalData['scale1'],newData['scale1']),axis=0)    
            if i%(batch_size-1)==0 and i!=0:
                scene_by_covisibility_score.append(finalData)
                finalData = {}
        list_of_scenes_by_covisibility_score.append(scene_by_covisibility_score)
    return list_of_scenes_by_covisibility_score