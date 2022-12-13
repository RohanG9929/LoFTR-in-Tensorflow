import tensorflow as tf
import tensorflow_addons as tfa
from einops.einops import rearrange
# print(tf.__version__)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return tf.keras.layers.Conv2D(out_planes,kernel_size=1,strides=stride,padding='valid',use_bias=False)

def conv3x3(in_planes, out_planes, stride=1,x=None):
    """3x3 convolution without padding
    our TF implementation will only pad tensors before passing into convolutional operation due to
    strict definition of paddings=['valid','same'] unlike torch which can accepts integers.
    """
    return tf.keras.layers.Conv2D(out_planes,kernel_size=3,strides=stride,padding='valid',use_bias=False)

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1=tf.keras.layers.BatchNormalization(axis=-1)
        self.bn2=tf.keras.layers.BatchNormalization(axis=-1)
        self.relu=tf.keras.layers.ReLU()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = tf.keras.Sequential([
                conv1x1(in_planes, planes, stride=stride),
                tf.keras.layers.BatchNormalization(axis=-1)
            ])

    def call(self, x):
        y = x
        y=tf.pad(y,tf.constant([[0,0],[1,1],[1,1],[0,0]]))
        # print(f'y shape after padding is {y.shape}')
        y = self.relu(self.bn1(self.conv1(y)))
        y=tf.pad(y,tf.constant([[0,0],[1,1],[1,1],[0,0]]))
        y = self.bn2(self.conv2(y))


        if self.downsample is not None:
            x = self.downsample(x)
        
        # print(f'the shape of x before relu is {x.shape}')
        # print(f'the shape of y before relu is {y.shape}')

        return self.relu(x+y)


class ResNetFPN_8_2(tf.keras.Model):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self,config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim'] # 128
        block_dims =config['block_dims']  # [128,196,256]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1=tf.keras.layers.Conv2D(initial_dim,kernel_size=7,strides=2,padding='valid',use_bias=False) # need tf.pad
        self.bn1=tf.keras.layers.BatchNormalization(axis=-1)
        self.relu=tf.keras.layers.ReLU()

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2= tf.keras.Sequential(
                            [conv3x3(block_dims[2], block_dims[2]),
                            tf.keras.layers.BatchNormalization(axis=-1),
                            tf.keras.layers.LeakyReLU(),
                            conv3x3(block_dims[2], 
                            block_dims[1])])
        
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2= tf.keras.Sequential(
                            [conv3x3(block_dims[1], block_dims[1]),
                            tf.keras.layers.BatchNormalization(axis=-1),
                            tf.keras.layers.LeakyReLU(),
                            conv3x3(block_dims[1], 
                            block_dims[0])])

        for m in self.layers:  
            if isinstance(m, tf.keras.layers.Conv2D):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                wt_initializer = tf.keras.initializers.HeNormal()
                m.kernel_initializer = wt_initializer
                #! no "fan out" mode available in "tf-HeNormal",  
                #! tf & pytorch functions used have different gains in their respective formulae
            elif isinstance(m, (tf.keras.layers.BatchNormalization, tfa.layers.GroupNormalization)):
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
                wt_initializer = tf.keras.initializers.Constant(1)
                m.kernel_initializer = wt_initializer
                bias_initializer = tf.keras.initializers.Constant(0)
                m.bias_regularizer = bias_initializer

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return tf.keras.Sequential([*layers])

    def call(self, x):
        x = rearrange(x, 'n c h w -> n h w c')
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0=tf.pad(x0,tf.constant([[0,0],[2,1],[2,1],[0,0]]))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        # x3_out = self.layer3_outconv(x3)
        # x2_out = self.layer2_outconv(x2)
        # x3_out_2x = tf.keras.layers.UpSampling2D(size=2, interpolation = 'bilinear', data_format='channels_last')(x3_out)
        # padding=tf.constant([[0,0],[2,2],[2,2],[0,0]])  
        # x2_out = self.layer2_outconv2(tf.pad(x2_out, padding, "CONSTANT")+ tf.pad(x3_out_2x, padding, "CONSTANT"))
        # x1_out = self.layer1_outconv(x1)
        # x2_out_2x = tf.keras.layers.UpSampling2D(size=2, interpolation = 'bilinear', data_format='channels_last')(x2_out)
        # x1_out = self.layer1_outconv2(tf.pad(x1_out, padding, "CONSTANT")+ tf.pad(x2_out_2x, padding, "CONSTANT"))

        x3_out = self.layer3_outconv(x3)
        x3_out_2x=tf.image.resize(x3_out,[x3_out.shape[1]*2,x3_out.shape[2]*2]) # should be [1, 256, 120, 160]
        x2_out = self.layer2_outconv(x2) # should be [1, 256, 120, 160]
        ################### fixing the shape issue due to having valid padding in conv3 ######################
        x2_out=tf.pad(x2_out,tf.constant([[0,0],[2,2],[2,2],[0,0]]))
        x3_out_2x=tf.pad(x3_out_2x,tf.constant([[0,0],[2,2],[2,2],[0,0]]))
        ###################################################################################################
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x) # pytorch shape: ([1, 196, 120, 160])
        x2_out_2x=tf.image.resize(x2_out,[x2_out.shape[1]*2,x2_out.shape[2]*2]) # pytorch shape: (1, 196, 240, 320])
        x1_out = self.layer1_outconv(x1) # pytorch shape: [1, 196, 240, 320])
        x1_out=tf.pad(x1_out,tf.constant([[0,0],[2,2],[2,2],[0,0]]))
        x2_out_2x=tf.pad(x2_out_2x,tf.constant([[0,0],[2,2],[2,2],[0,0]]))
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x) # pytorch shape: [1, 128, 240, 320])

        x3_out = rearrange((x3_out), 'n h w c -> n c h w')
        x1_out = rearrange((x1_out), 'n h w c -> n c h w')
        return [x3_out, x1_out]


############## below just for testing########################################
# input_shape=(2,1,480,640) # (batch,height,width,number of channels)
# image=tf.random.normal(input_shape)

# config={'initial_dim':128,'block_dims':[128,196,256]}
# model_final=ResNetFPN_8_2(config)
# ans_final=model_final.call(image)
# print(f'the shape of x3 final is {ans_final[0].shape}') # torch answer torch.Size([1, 256, 60, 80])
# print(f'the shape of x1 final is {ans_final[1].shape}') # torch answer torch.Size([1, 128, 240, 320])
################################################################################