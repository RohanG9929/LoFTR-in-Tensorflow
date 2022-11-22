import  tensorflow as tf
from einops.einops import rearrange

# from .backbone import build_backbone
from Module1 import Module1
from posEncode_TF import PositionEncodingSine
from transformer_TF import LocalFeatureTransformer
from fine_processTF import FinePreprocess
from coarse_matchingTF import CoarseMatching
from fine_matchingTF import FineMatching


class LoFTR(tf.keras.Model):
    def __init__(self, config):
        super(LoFTR,self).__init__()
        # Misc
        self.config = config

        # Modules
        # self.backbone = Resnet_FPN(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

    def call(self, data, training=False):
        """ 
        Update:
            data (dict): {
                'image0': (tf.Tensor): (N, 1, H, W)
                'image1': (tf.Tensor): (N, 1, H, W)
                'mask0'(optional) : (tf.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (tf.Tensor): (N, H, W)
            }
        """
        ################################################################################################################
        # 1. Local Feature CNN
        ################################################################################################################
        data.update({
            'bs': data['image0'].shape[0], #Batch size BS
            'hw0_i': data['image0'].shape[2:], #Shape of image 1
            'hw1_i': data['image1'].shape[2:] #Shape of image 2
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            #Course and fine feature maps
            #c is 1/8
            #f is 1/2
            # feats_c, feats_f = self.backbone(tf.concat([data['image0'], data['image1']], axis=0))
            feats_c, feats_f = Module1(tf.concat([data['image0'], data['image1']], axis=0))
    
            #Splitting into features A and features B (feat im1 and feat im2)
            #For course and fine
            (feat_c0, feat_c1) = tf.split(feats_c,num_or_size_splits=(2),axis=0)
            (feat_f0, feat_f1) = tf.split(feats_f,num_or_size_splits=(2),axis=0)
        else:  # handle different input shapes
            # (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])
            (feat_c0, feat_f0), (feat_c1, feat_f1) = Module1(data['image0']), Module1(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:], 
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })
        print('Module 1 Done')
        
        ################################################################################################################
        # 2. Coarse-level loftr module
        ################################################################################################################
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        # if 'mask0' in data:
        #     mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2) 
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        print('Module 2 Done')

        ################################################################################################################
        # 3. Match coarse-level
        ################################################################################################################
        data = self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1, training=training)
        print('Module 3 Done')

        ################################################################################################################
        # 4. Fine-level refinement
        ################################################################################################################
        feat_f0_unfold, feat_f1_unfold, data = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.shape[0] != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        print('Module 4 Done')

        ################################################################################################################
        # 5. Match fine-level
        ################################################################################################################
        data = self.fine_matching(feat_f0_unfold, feat_f1_unfold, data, training = training)
        print('Module 5 Done')
        return data


