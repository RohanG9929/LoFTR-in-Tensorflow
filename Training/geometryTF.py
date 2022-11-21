import tensorflow as tf
import numpy as np


#@tf.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (tf.Tensor): [N, L, 2] - <x, y>,
        depth0 (tf.Tensor): [N, H, W],
        depth1 (tf.Tensor): [N, H, W],
        T_0to1 (tf.Tensor): [N, 3, 4],
        K0 (tf.Tensor): [N, 3, 3],
        K1 (tf.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (tf.Tensor): [N, L]
        warped_keypoints0 (tf.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """

    kpts0_long = tf.cast(tf.math.round(kpts0),dtype=tf.float32)

    # Sample depth, get calculable_mask on depth != 0
   
    kpts0_depth = np.zeros((kpts0.shape[0],4800))
    for i in range(kpts0.shape[0]):
        for a in range(kpts0_long.shape[1]):
            kpts0_depth[i,a] = depth0[i, int(kpts0_long[i, a, 1].numpy()), int(kpts0_long[i, a, 0].numpy())]
    kpts0_depth = tf.convert_to_tensor(kpts0_depth,dtype=tf.float32)
    # kpts0_depth = tf.stack(
    #     [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], axis=0)  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    temp = tf.reshape(tf.ones_like(kpts0[:, :, 0]),[tf.ones_like(kpts0[:, :, 0]).shape[0],tf.ones_like(kpts0[:, :, 0]).shape[1],1])
    kpts0_h = tf.concat([kpts0, temp], axis=-1).numpy() * kpts0_depth[..., None].numpy()  # (N, L, 3)
    kpts0_h = tf.convert_to_tensor(kpts0_h)

    kpts0_cam = tf.linalg.inv(K0) @ tf.transpose(kpts0_h,[0,2,1])  # (N, 3, L)

    # Rigid Transform
    tempT_0to1 = T_0to1.numpy()
    w_kpts0_cam = T_0to1[:, :3, :3].numpy() @ kpts0_cam.numpy() + tempT_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_cam = tf.convert_to_tensor(w_kpts0_cam)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = tf.transpose((K1 @ w_kpts0_cam),[0,2,1])  # (N, L, 3)
    tempw_kpts0_h = w_kpts0_h.numpy()
    w_kpts0 = w_kpts0_h[:, :, :2].numpy() / (tempw_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth
    w_kpts0 = tf.convert_to_tensor(w_kpts0)

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = tf.cast((w_kpts0[:, :, 0] > 0) ,dtype=tf.int32)* tf.cast((w_kpts0[:, :, 0] < w-1),dtype=tf.int32) * tf.cast((w_kpts0[:, :, 1] > 0) ,dtype=tf.int32)* tf.cast((w_kpts0[:, :, 1] < h-1),dtype=tf.int32)
    w_kpts0_long = tf.cast(w_kpts0,tf.float32)
    w_kpts0_long = w_kpts0_long.numpy()
    w_kpts0_long[~covisible_mask, :] = 0
    w_kpts0_long = tf.convert_to_tensor(w_kpts0_long)



       
    w_kpts0_depth = np.zeros((kpts0.shape[0],4800))
    for i in range(w_kpts0_long.shape[0]):
        for a in range(w_kpts0_long.shape[1]):
            w_kpts0_depth[i,a] = depth1[i, int(kpts0_long[i, a, 1].numpy()), int(kpts0_long[i, a, 0].numpy())]
    w_kpts0_depth = tf.convert_to_tensor(w_kpts0_depth,dtype=tf.float64) # (N, L)

    # w_kpts0_depth = tf.stack(
    #     [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], axis=0)  # (N, L)

    consistent_mask = tf.math.abs((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth) < 0.2
    valid_mask = nonzero_mask.numpy() * tf.cast(covisible_mask,dtype=tf.bool).numpy() * consistent_mask.numpy()
    valid_mask = tf.convert_to_tensor(valid_mask)

    return valid_mask, w_kpts0
