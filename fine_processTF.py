
import tensorflow as tf
from einops.einops import rearrange, repeat


class FinePreprocess(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj =  tf.keras.layers.Dense(d_model_f, activation=None)
            # self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat =  tf.keras.layers.Dense(d_model_f, activation=None)
            # self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def call(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            # feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            # feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat0 = tf.zeros([0, self.W**2, self.d_model_f])
            feat1 = tf.zeros([0, self.W**2, self.d_model_f])
            return feat0, feat1, data

        # 1. unfold(crop) all local windows
        feat_f0_unfold = tf.image.extract_patches(tf.transpose(feat_f0,[0,2,3,1]), sizes=[1,W,W,1], strides=[1,stride,stride,1],rates=[1,1,1,1], padding='SAME')
        feat_f0_unfold = tf.reshape(feat_f0_unfold,(feat_f0_unfold.shape[0],feat_f0_unfold.shape[1]*feat_f0_unfold.shape[2],feat_f0_unfold.shape[3]))
        feat_f0_unfold = tf.transpose(feat_f0_unfold,[0,2,1])
        # feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        feat_f1_unfold = tf.image.extract_patches(tf.transpose(feat_f1,[0,2,3,1]), sizes=[1,W,W,1], strides=[1,stride,stride,1],rates=[1,1,1,1], padding='SAME')
        feat_f1_unfold = tf.reshape(feat_f1_unfold,(feat_f1_unfold.shape[0],feat_f1_unfold.shape[1]*feat_f1_unfold.shape[2],feat_f1_unfold.shape[3]))
        feat_f1_unfold = tf.transpose(feat_f1_unfold,[0,2,1])
        # feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold.numpy()[data['b_ids'].numpy(), data['i_ids'].numpy()]  # [n, ww, cf]
        feat_f0_unfold = tf.convert_to_tensor(feat_f0_unfold)

        feat_f1_unfold = feat_f1_unfold.numpy()[data['b_ids'].numpy(), data['j_ids'].numpy()]
        feat_f1_unfold = tf.convert_to_tensor(feat_f1_unfold)
        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(tf.concat([feat_c0.numpy()[data['b_ids'].numpy(), data['i_ids'].numpy()],
                                                   feat_c1.numpy()[data['b_ids'].numpy(), data['j_ids'].numpy()]], axis=0))  # [2n, c]
            feat_cf_win = self.merge_feat(tf.concat([
                tf.concat([feat_f0_unfold, feat_f1_unfold], axis=0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
            ], -1))
            # feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)
            feat_f0_unfold, feat_f1_unfold =  tf.split(feat_cf_win,num_or_size_splits=2,axis=0)

        return feat_f0_unfold, feat_f1_unfold, data
