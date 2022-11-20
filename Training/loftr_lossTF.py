from loguru import logger

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np


class LoFTRLoss(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftr']['loss']
        self.match_type = self.config['loftr']['match_coarse']['match_type']
        self.sparse_spvs = self.config['loftr']['match_coarse']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        #if not pos_mask.any():  # assign a wrong gt
        if not tf.reduce_any(pos_mask):
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not tf.reduce_any(neg_mask):
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            #conf = torch.clamp(conf, 1e-6, 1-1e-6)
            conf=tf.clip_by_value(conf, clip_value_min=1e-6, clip_value_max=1-1e-6)
            #loss_pos = - torch.log(conf[pos_mask])
            #loss_neg = - torch.log(1 - conf[neg_mask])
            loss_pos=-tf.math.log(conf[pos_mask])
            loss_neg=-tf.math.log(1-conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = tf.clip_by_value(conf,clip_value_min=1e-6, clip_value_max=1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * tf.math.pow(1 - pos_conf, gamma) * tf.math.log(pos_conf)
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = tf.math_reduce_sum(conf_gt,axis=-1) == 0, tf.math_reduce_sum(conf_gt,axis=1) == 0
                    #neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    neg_conf = tf.concat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], axis=0, name='concat')
                    loss_neg = - alpha * tf.math.pow(1 - neg_conf, gamma) * tf.math.log(neg_conf)
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (tf.math.reduce_sum(weight,axis=-1) != 0)[neg0]
                        neg_w1 = (tf.math.reduce_sum(weight,axis=1) != 0)[neg1]
                        neg_mask = tf.concat([neg_w0, neg_w1], axis=0,name='concat')
                        loss_neg = loss_neg[neg_mask]
                
                loss =  c_pos_w * tf.math.reduce_mean(loss_pos) + c_neg_w * tf.math.reduce_mean(loss_neg) \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * tf.math.reduce_mean(loss_pos)
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * tf.math.pow(1 - conf[pos_mask], gamma) * tf.math.log(conf[pos_mask])
                loss_neg = - alpha * tf.math.pow(conf[neg_mask], gamma) * tf.math.log(1 - conf[neg_mask])
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * tf.math.reduce_mean(loss_pos) + c_neg_w * tf.math.reduce_mean(loss_neg)
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_fine_loss(self, expec_f, expec_f_gt,training):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt,training)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt,training)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt,training):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = tf.norm(expec_f_gt, ord=np.inf, axis=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        #offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        offset_l2=tf.math.reduce_sum((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2,axis=-1)
        return tf.math.reduce_mean(offset_l2)

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt, training):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = tf.norm(expec_f_gt, ord=np.inf, axis=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / tf.clip_by_value(std, clip_value_min=1e-10,clip_value_max=float('inf'))
        weight = (inverse_std / tf.math.reduce_mean(inverse_std))#.detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.numpy().any():
            if training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        #offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        offset_l2=tf.math.reduce_sum((expec_f_gt.numpy()[correct_mask.numpy()] - expec_f.numpy()[correct_mask.numpy(), :2]) ** 2,axis=-1)

        #loss = (offset_l2 * weight[correct_mask]).mean()
        loss=tf.math.reduce_mean(offset_l2 * weight[correct_mask])

        return loss
    
    #@torch.no_grad()
    # @tf.stop_gradient()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            #c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
            c_weight=tf.reshape(data['mask0'],(data['mask0'].shape[0]*data['mask0'].shape[1]))[..., None]*\
                tf.reshape(data['mask1'],(data['mask1'].shape[0]*data['mask1'].shape[1]))[..., None]
            c_weight=tf.cast( c_weight,tf.float32)
        else:
            c_weight = None
        return c_weight

    def call(self, data, training = True):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight)
        loss = loss_c * self.loss_config['coarse_weight']
        #loss_scalars.update({"loss_c": loss_c.clone()})#.detach().cpu()})
        loss_scalars.update({"loss_c": tf.identity(loss_c)})#.detach().cpu()})


        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'],training)
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            #loss_scalars.update({"loss_f":  loss_f.clone()})#.detach().cpu()})
            loss_scalars.update({"loss_f": tf.identity(loss_f)})#.detach().cpu()})
        else:
            assert training is False
            loss_scalars.update({'loss_f': tf.tensor(1.)})  # 1 is the upper bound

        #loss_scalars.update({'loss': loss.clone()})#.detach().cpu()})
        loss_scalars.update({"loss": tf.identity(loss)})#.detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
        print('Loss Done')
        return data
