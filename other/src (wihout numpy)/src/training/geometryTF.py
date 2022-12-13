import tensorflow as tf


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

    kpts0_long = tf.cast(tf.math.round(kpts0),dtype=tf.int64)

    # Sample depth, get calculable_mask on depth != 0

    kpts0_depth = tf.stack(
        [tf.gather_nd(depth0,indices=[[x,y,z] for x,y,z in zip(*[i*tf.ones(kpts0_long.shape[1],dtype=tf.int64), kpts0_long[i,:, 1], kpts0_long[i, :, 0]])])
        for i in range(kpts0.shape[0])], axis=0)# (N, L)

    nonzero_mask = kpts0_depth != 0

    # Unproject
    temp = tf.reshape(tf.ones_like(kpts0[:, :, 0]),[tf.ones_like(kpts0[:, :, 0]).shape[0],tf.ones_like(kpts0[:, :, 0]).shape[1],1])
    kpts0_h = tf.cast(tf.concat([kpts0, temp], axis=-1),tf.float32)* kpts0_depth[..., None] # (N, L, 3)
    kpts0_cam = tf.linalg.inv(K0) @ tf.transpose(kpts0_h,[0,2,1])  # (N, 3, L)

    # Rigid Transform
    tempT_0to1 = tf.gather_nd(T_0to1,indices=[[[i,0,3],[i,1,3],[i,2,3]] for i in range(T_0to1.shape[0])])
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam+ tf.reshape(tempT_0to1,[tempT_0to1.shape[0],tempT_0to1.shape[1],1])   # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = tf.transpose((K1 @ w_kpts0_cam),[0,2,1])  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (tf.gather(w_kpts0_h,indices=[2],axis=2) + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = tf.cast((w_kpts0[:, :, 0] > 0) ,dtype=tf.int32) * \
                    tf.cast((w_kpts0[:, :, 0] < w-1),dtype=tf.int32) * \
                    tf.cast((w_kpts0[:, :, 1] > 0) ,dtype=tf.int32) * \
                    tf.cast((w_kpts0[:, :, 1] < h-1),dtype=tf.int32)
    w_kpts0_long = tf.cast(w_kpts0,tf.int64)
    covisible_mask = tf.cast(covisible_mask,tf.bool)
    indexes = tf.where(~covisible_mask)
    w_kpts0_long = tf.stack([tf.tensor_scatter_nd_update(w_kpts0_long[:,:,0],indexes, tf.zeros(indexes.shape[0],dtype=tf.int64)),
            tf.tensor_scatter_nd_update(w_kpts0_long[:,:,1],indexes, tf.zeros(indexes.shape[0],dtype=tf.int64))],axis=2)

    # w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = tf.stack(
        [tf.gather_nd(depth1,indices=[[x,y,z] for x,y,z in zip(*[i*tf.ones(w_kpts0_long.shape[1],dtype=tf.int64), w_kpts0_long[i,:, 1], w_kpts0_long[i, :, 0]])])
        for i in range(w_kpts0_long.shape[0])], axis=0)# (N, L)

  

    consistent_mask = tf.math.abs((tf.cast(w_kpts0_depth,tf.double) - tf.cast(w_kpts0_depth_computed,tf.double)) / tf.cast(w_kpts0_depth,tf.double)) < 0.2
    valid_mask = tf.cast(tf.cast(nonzero_mask,tf.int64) * tf.cast(covisible_mask,dtype=tf.int64) * tf.cast(consistent_mask,dtype=tf.int64),tf.bool)

    return valid_mask, tf.cast(w_kpts0,tf.float64)



