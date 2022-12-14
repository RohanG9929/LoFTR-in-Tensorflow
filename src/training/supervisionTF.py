from math import log
from loguru import logger

import tensorflow as tf
from einops import repeat

from .geometryTF import warp_kpts

def create_mesh(height: int,width: int,normalized_coordinates: bool = True):

    xs = tf.linspace(0, width - 1, width)
    ys = tf.linspace(0, height - 1, height)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates

    base_grid = tf.stack(tf.meshgrid(xs, ys, indexing="ij"), axis=-1, name='stack')  # WxHx2

    return  tf.expand_dims(tf.transpose(base_grid,[1,0,2]), axis=0)

##############  ↓  Coarse-Level supervision  ↓  ##############


# @torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


# @torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    # device = data['image0'].device
    N,_, H0, W0 = data['image0'].shape
    _,_, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_mesh(h0, w0, False)
    grid_pt0_c = tf.reshape(grid_pt0_c,[1, h0*w0, 2])
    grid_pt0_c = tf.repeat(grid_pt0_c,N,axis=0)    # [N, hw, 2]

    grid_pt0_i = tf.cast(scale0,tf.double) * grid_pt0_c
    grid_pt1_c = create_mesh(h1, w1, False)
    grid_pt1_c = tf.reshape(grid_pt1_c,[1, h1*w1, 2])
    grid_pt1_c = tf.repeat(grid_pt1_c,N,axis=0)
    grid_pt1_i = tf.cast(scale1,tf.double) * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])


    minVal_1 = tf.math.minimum(tf.cast(data['depth0'].shape[1]-1,tf.double),tf.cast(data['depth0'].shape[2]-1,tf.double))
    minVal_2 = tf.math.minimum(tf.cast(data['depth1'].shape[1]-1,tf.double),tf.cast(data['depth1'].shape[2]-1,tf.double))
    grid_pt0_i = tf.clip_by_value(grid_pt0_i, 0, minVal_1, name=None)
    grid_pt1_i = tf.clip_by_value(grid_pt1_i, 0, minVal_2, name=None)
    
    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = tf.cast(tf.math.round(w_pt0_c[:, :, :]),dtype=tf.int64)
    
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = tf.cast(tf.math.round(w_pt1_c[:, :, :]),dtype=tf.int64)
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return tf.convert_to_tensor((pt[..., 0] < 0).numpy() + (pt[..., 0] >= w).numpy() + (pt[..., 1] < 0).numpy() + (pt[..., 1] >= h).numpy())
  
    nearest_index1 = nearest_index1.numpy()
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1).numpy()] = 0
    nearest_index1 = tf.convert_to_tensor(nearest_index1)

    nearest_index0 = nearest_index0.numpy()
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0).numpy()] = 0
    nearest_index0 = tf.convert_to_tensor(nearest_index0)

    loop_back = tf.stack([nearest_index0.numpy()[_b][tf.cast(_i,tf.int32)] for _b, _i in enumerate(nearest_index1)],axis=0) # (N, L)


    # import numpy as np
    # loop_back = np.zeros((nearest_index1.shape[0],nearest_index1.shape[1]))
    # for _b, _i in enumerate(nearest_index1):
    #     for a in range(_i.shape[0]):
    #             loop_back[_b,int(_i[a])] = nearest_index0[_b][int(_i[a])]
    # loop_back = tf.convert_to_tensor(loop_back)

    correct_0to1 = loop_back == tf.repeat(tf.range(h0*w0)[None],N,axis=0) 
    correct_0to1 = correct_0to1.numpy()
    correct_0to1[:, 0] = False  # ignore the top-left corner
    correct_0to1 = tf.convert_to_tensor(correct_0to1)



    # 4. construct a gt conf_matrix
    conf_matrix_gt = tf.zeros([N, h0*w0, h1*w1])
    temp = tf.where(correct_0to1 != False)
    b_ids, i_ids = temp[:,0], temp[:,1]
    j_ids = tf.convert_to_tensor(nearest_index1.numpy()[b_ids.numpy(),i_ids.numpy()],dtype=tf.int64)#tf.gather_nd(tf.cast(nearest_index1,tf.int64), list(zip(b_ids,i_ids)))

    
    conf_matrix_gt = conf_matrix_gt.numpy()
    conf_matrix_gt[b_ids.numpy().astype(int), i_ids.numpy().astype(int), j_ids.numpy().astype(int)] = 1
    conf_matrix_gt = tf.convert_to_tensor(conf_matrix_gt)
    # conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found")
        # this won't affect fine-level loss calculation
        b_ids = tf.constant([0])
        i_ids = tf.constant([0])
        j_ids = tf.constant([0])

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })
    # print("Course SuperVision Done")
    return data


def compute_supervision_coarse(data, config):
    return spvs_coarse(data, config)

    # assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    # data_source = data['dataset_name'][0]
    # if data_source.lower() in ['scannet', 'megadepth']:
    #     data = spvs_coarse(data, config)
    # else:
    #     raise ValueError(f'Unknown data source: {data_source}')
    # return data


##############  ↓  Fine-Level supervision  ↓  ##############

# @torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['LOFTR']['RESOLUTION'][1]
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    scale = scale * data['scale1'].numpy()[b_ids.numpy()] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i.numpy()[b_ids.numpy(), i_ids.numpy()] - pt1_i.numpy()[b_ids.numpy(), tf.cast(j_ids,tf.int64).numpy()]) / scale / radius  # [M, 2]
    expec_f_gt = tf.convert_to_tensor(expec_f_gt)
    data.update({"expec_f_gt": expec_f_gt})
    return data


def compute_supervision_fine(data, config):
    # print("Fine supervision Done")
    return spvs_fine(data, config)

    # data_source = data['dataset_name'][0]
    # if data_source.lower() in ['scannet', 'megadepth']:
    #     data = spvs_fine(data, config)
    # else:
    #     raise NotImplementedError
    # return data
