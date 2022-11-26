import math
import tensorflow as tf


class PositionEncodingSine(tf.keras.Model):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        self.pe = tf.zeros((d_model, *max_shape))
        #y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        y_position = tf.ones(max_shape)
        y_position =  tf.cast(y_position, tf.float32)
        y_position = tf.expand_dims(y_position, axis=0)

        # x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        x_position = tf.ones(max_shape)
        x_position =  tf.cast(x_position, tf.float32)
        x_position = tf.expand_dims(x_position, axis=0)

        if temp_bug_fix:
            div_term = tf.range(0, d_model//2, 2)
            div_term = tf.cast(div_term, tf.float32) * (-math.log(10000.0) / (d_model//2))
            div_term = tf.exp(div_term)
            # div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = tf.range(0, d_model//2, 2)
            div_term = tf.cast(div_term, tf.float32) * (-math.log(10000.0) / (d_model//2))
            div_term = tf.exp(div_term)
            # div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))

        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        
        # pe[0::4, :, :] = tf.math.sin(x_position * div_term)
        # pe[1::4, :, :] = tf.math.cos(x_position * div_term)
        # pe[2::4, :, :] = tf.math.sin(y_position * div_term)
        # pe[3::4, :, :] = tf.math.cos(y_position * div_term)
        a = tf.math.sin(x_position * div_term)
        b = tf.math.cos(x_position * div_term)
        c = tf.math.sin(y_position * div_term)
        d = tf.math.cos(y_position * div_term)
        self.pe = tf.concat([a,b,c,d],0)
        myPE = tf.expand_dims(self.pe,0)
        # self.register_buffer('pe', myPE, persistent=False)  # [1, C, H, W]
        self.pe = tf.reshape(self.pe, (1,self.pe.shape[0],self.pe.shape[1],self.pe.shape[2]))
    def call(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.shape[2], :x.shape[3]]
