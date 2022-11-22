from einops.einops import rearrange
import tensorflow as tf
import numpy as np

INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (tf.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    np_m = m.numpy()

    np_m[:, :b] = v
    np_m[:, :, :b] = v
    np_m[:, :, :, :b] = v
    np_m[:, :, :, :, :b] = v
    np_m[:, -b:] = v
    np_m[:, :, -b:] = v
    np_m[:, :, :, -b:] = v
    np_m[:, :, :, :, -b:] = v

    m = tf.convert_to_tensor(np_m)

    return m
    # tensor = m
    # indices = tf.constant([[:, :b]], [[[:, :, :b]]], [[[[:, :, :, :b]]]], [[[[[:, :, :, :, :b]]]]], [[:, -b:]], [[[:, :, -b:]]], [[[[:, :, :, -b:]]]], [[[[[:, :, :, :, -b:]]]]])
    # updates = tf.constant(v,v,v,v,v,v,v,v)
    # tf.tensor_scatter_nd_update(tensor, indices, updates)



def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    np_m = m.numpy()
    np_m[:, :bd] = v
    np_m[:, :, :bd] = v
    np_m[:, :, :, :bd] = v
    np_m[:, :, :, :, :bd] = v

    # h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h0s, w0s = tf.cast(tf.math.reduce_max(tf.math.reduce_sum(p_m0, 1),-1)[0], tf.int32), tf.cast(tf.math.reduce_max(tf.math.reduce_sum(p_m0, -1),-1)[0], tf.int32)
    # h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    h1s, w1s = tf.cast(tf.math.reduce_max(tf.math.reduce_sum(p_m1, 1),-1)[0], tf.int32), tf.cast(tf.math.reduce_max(tf.math.reduce_sum(p_m1, -1),-1)[0], tf.int32)

    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        np_m[b_idx, h0 - bd:] = v
        np_m[b_idx, :, w0 - bd:] = v
        np_m[b_idx, :, :, h1 - bd:] = v
        np_m[b_idx, :, :, :, w1 - bd:] = v
    
    m = tf.convert_to_tensor(np_m)
    return m

def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (tf.Tensor): padded masks
    """
    # h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h0s, w0s = tf.math.reduce_max(tf.math.reduce_sum(p_m0, 1),-1)[0], tf.math.reduce_max(tf.math.reduce_sum(p_m0, -1),-1)[0]
    # h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    h1s, w1s = tf.math.reduce_max(tf.math.reduce_sum(p_m1, 1),-1)[0], tf.math.reduce_max(tf.math.reduce_sum(p_m1, -1),-1)[0]

    # max_cand = tf.sum(
    #     tf.min(tf.stack([h0s * w0s, h1s * w1s], -1), -1)[0])

    max_cand = tf.math.reduce_sum(tf.math.reduce_min(tf.stack([h0s * w0s, h1s * w1s], -1), -1)[0])

    return max_cand


class CoarseMatching(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        # elif self.match_type == 'sinkhorn':
            # try:
            #     from .superglue import log_optimal_transport
            # except ImportError:
            #     raise ImportError("download superglue.py first!")
            # self.log_optimal_transport = log_optimal_transport
            # self.bin_score = nn.Parameter(
            #     tf.tensor(config['skh_init_bin_score'], requires_grad=True))
            # self.skh_iters = config['skh_iters']
            # self.skh_prefilter = config['skh_prefilter']
        # else:
        #     raise NotImplementedError()

    def call(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None, training = False):
        """
        Args:
            feat0 (tf.Tensor): [N, L, C]
            feat1 (tf.Tensor): [N, S, C]
            data (dict)
            mask_c0 (tf.Tensor): [N, L] (optional)
            mask_c1 (tf.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (tf.Tensor): [M'],
                'i_ids' (tf.Tensor): [M'],
                'j_ids' (tf.Tensor): [M'],
                'gt_mask' (tf.Tensor): [M'],
                'mkpts0_c' (tf.Tensor): [M, 2],
                'mkpts1_c' (tf.Tensor): [M, 2],
                'mconf' (tf.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.get_shape().as_list()[0], feat_c0.get_shape().as_list()[1], feat_c1.get_shape().as_list()[1], feat_c0.get_shape().as_list()[2]

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.get_shape().as_list()[-1]**.5,
                               [feat_c0, feat_c1])

        if self.match_type == 'dual_softmax':
            sim_matrix = tf.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature
            if mask_c0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)
            conf_matrix = tf.nn.softmax(sim_matrix, 1) * tf.nn.softmax(sim_matrix, 2)

        # elif self.match_type == 'sinkhorn':
        #     # sinkhorn, dustbin included
        #     sim_matrix = tf.einsum("nlc,nsc->nls", feat_c0, feat_c1)
        #     if mask_c0 is not None:
        #         sim_matrix[:, :L, :S].masked_fill(
        #             ~tf.cast(mask_c0[..., None] * mask_c1[:, None],tf.bool),
        #             -INF)

            # build uniform prior & use sinkhorn
            # log_assign_matrix = self.log_optimal_transport(
            #     sim_matrix, self.bin_score, self.skh_iters)
            # assign_matrix = log_assign_matrix.exp()
            # conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            # if not training and self.skh_prefilter:
            #     filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
            #     filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
            #     conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
            #     conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            # if self.config['sparse_spvs']:
            #     data.update({'conf_matrix_with_bin': assign_matrix.clone()})

        data.update({'conf_matrix': conf_matrix})
        
        # predict coarse matches from conf_matrix
        # self.get_coarse_match(conf_matrix, data)
        data.update(**self.get_coarse_match(conf_matrix, data, training))
        return data

    # @tf.no_grad()
    def get_coarse_match(self, conf_matrix, data, training):
        """
        Args:
            conf_matrix (tf.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (tf.Tensor): [M'],
                'i_ids' (tf.Tensor): [M'],
                'j_ids' (tf.Tensor): [M'],
                'gt_mask' (tf.Tensor): [M'],
                'm_bids' (tf.Tensor): [M],
                'mkpts0_c' (tf.Tensor): [M, 2],
                'mkpts1_c' (tf.Tensor): [M, 2],
                'mconf' (tf.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask = mask_border(mask, self.border_rm, False)
        else:
            mask = mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # 2. mutual nearest
        # mask = mask \
        #     * (conf_matrix == conf_matrix.max(axis=2, keepdim=True)[0]) \
        #     * (conf_matrix == conf_matrix.max(axis=1, keepdim=True)[0])
        b1 = (conf_matrix == tf.math.reduce_max(conf_matrix, axis=2, keepdims=True)[0])
        b2 =  (conf_matrix == tf.math.reduce_max(conf_matrix, axis=1, keepdims=True)[0])
        b3 = tf.math.logical_and(b1,b2)
        mask = tf.math.logical_and(mask, b3)


        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        # mask_v, all_j_ids = tf.math.reduce_max(mask, axis=2)
        # b_ids, i_ids = tf.where(mask_v)
        # j_ids = all_j_ids[b_ids, i_ids]
        # mconf = conf_matrix[b_ids, i_ids, j_ids]
        mask_int = tf.cast(mask, dtype=tf.int32)
        mask_v_int = tf.reduce_max(mask_int, axis=2)
        mask_v = tf.cast(mask_v_int, dtype=tf.bool)
        all_j_ids = tf.math.argmax(mask, axis=2)
        temp = tf.where(mask_v)
        b_ids, i_ids = temp[:,0],temp[:,1]
        if ((tf.size(b_ids) and tf.size(i_ids)) == 0):
            j_ids = tf.experimental.numpy.empty(0)
            mconf = tf.experimental.numpy.empty(0)
        else:
            j_ids = tf.gather_nd(all_j_ids,list(zip(b_ids,i_ids)))
            mconf =  tf.gather_nd(conf_matrix ,list(zip(b_ids,i_ids,j_ids)))

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.shape[0] * max(mask.shape[1], mask.shape[2])
            else:
                num_candidates_max = compute_max_candidates(data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                # pred_indices = tf.arange(num_matches_pred, device=_device)
                pred_indices = tf.range(num_matches_pred)
            else:
                # pred_indices = tf.randint(
                #     num_matches_pred,
                #     (num_matches_train - self.train_pad_num_gt_min, ),
                #     device=_device)
                pred_indices = tf.random.uniform(
                    (num_matches_train - self.train_pad_num_gt_min, ), maxval = num_matches_pred, dtype = tf.int32)              

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            # gt_pad_indices = tf.randint(len(data['spv_b_ids']),(max(num_matches_train - num_matches_pred,self.train_pad_num_gt_min), ),device=_device)
            # mconf_gt = tf.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            gt_pad_indices = tf.random.uniform(
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),  maxval = len(data['spv_b_ids']), dtype =tf.int32)
            mconf_gt = tf.zeros(len(data['spv_b_ids']))  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(lambda x, y: tf.concat([x.numpy()[pred_indices.numpy()], y.numpy()[gt_pad_indices.numpy()]],axis=0),
                                         *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],[j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['image0'].shape[2:][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c_np = np.stack(
            [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
            axis=1) * scale0
        mkpts1_c_np = np.stack(
            [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
            axis=1) * scale1
        mkpts0_c = tf.convert_to_tensor(mkpts0_c_np)
        mkpts1_c = tf.convert_to_tensor(mkpts1_c_np)


        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches
