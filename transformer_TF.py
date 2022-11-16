import copy
from linear_attention_TF import LinearAttention, FullAttention
import tensorflow as tf

class LoFTREncoderLayer(tf.keras.Model):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = tf.keras.layers.Dense(d_model, input_shape=(d_model,), activation=None) 
        self.k_proj = tf.keras.layers.Dense(d_model, input_shape=(d_model,), activation=None) 
        self.v_proj = tf.keras.layers.Dense(d_model, input_shape=(d_model,), activation=None) 

        # self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = tf.keras.layers.Dense(d_model, input_shape=(d_model,), activation=None) 

        # self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model*2, activation=None),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(d_model, activation=None),

        ])
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model*2, d_model*2, bias=False),
        #     nn.ReLU(True),
        #     nn.Linear(d_model*2, d_model, bias=False),
        # )


        # norm and dropout
        self.norm1 = tf.keras.layers.LayerNormalization(input_shape=(d_model,))
        self.norm2 = tf.keras.layers.LayerNormalization(axis=2)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)

    def call(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.shape[0]
        query, key, value = x, source, source

        # multi-head attention
        # query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        query = tf.reshape(self.q_proj(query), (bs, -1, self.nhead, self.dim)) # [N, L, (H, D)]

        # key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        key = tf.reshape(self.k_proj(key), (bs, -1, self.nhead, self.dim))

        # value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        value = tf.reshape(self.v_proj(value), (bs, -1, self.nhead, self.dim))

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        # message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.merge(tf.reshape(message, (bs, -1, self.nhead*self.dim)))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(tf.concat([x, message], axis=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(tf.keras.Model):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self,config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']


        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])

        self.myLayers = [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]


    def call(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"

        for layer, name in zip(self.myLayers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
